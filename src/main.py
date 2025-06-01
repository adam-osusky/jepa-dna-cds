import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from src.jepa import train_jepa
from src.window import create_classification_dataset
from src.classify import train_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gb_path",
        default="data/E_coli_K12.gbff",
        help="Path to the input GenBank (.gbff/.gb) file.",
    )
    parser.add_argument(
        "--classification_ds",
        default="data/classification_ds.csv",
        help="Path to the output CSV file (default: %(default)s).",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=512,
        help="Length of each sliding window (default: %(default)s).",
    )
    parser.add_argument(
        "-s",
        "--step_size",
        type=int,
        default=128,
        help="Step size overlap between windows (default: %(default)s).",
    )
    parser.add_argument(
        "-m",
        "--min_overlap_fraction",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of window overlapping a CDS to label it as coding. "
            "Value between 0 and 1 (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=69,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of train passes through the data.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for training."
    )
    parser.add_argument(
        "--ema_tau",
        type=float,
        default=0.95,
        help="In-place EMA update of target's parameters",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.15,
        help="Fraction of bp positions to mask during JEPA x transformation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num of parallel workers",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,
        help="Hidden dimension in transformer layers in encoder.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=16*4,
        help="Feedforward dimension size in transformer layers in encoder.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=2,
        help="Number of attention heads in transformer layers in encoder.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of transformer layers in encoder.",
    )
    parser.add_argument(
        "--wandb_logging",
        action="store_true",
        # default=False,
        default=True,
        help="Enable Weights & Biases logging of metrics.",
    )

    parser.add_argument(
        "--run_classification",
        action="store_true",
        # default=False,
        default=True,
        help="After JEPA pretrain, train a classifier (frozen-encoder or raw-sequence).",
    )
    parser.add_argument(
        "--pretrained_enc",
        # default=None,
        default="data/experiments/stellar-morning-11/dna_encoder_jepa.pth",
        help="Path to the pretrained encoder (default: %(default)s)."
        "If not specified will use classification head on new learnable embeddings",
    )
    parser.add_argument(
        "--clf_lr",
        type=float,
        default=0.001,
        help="Learning rate for classification head.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=1000,
        help="Number of epochs to train the classification head.",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=1,
        help="Batch size for classification training.",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        # default=4,
        default=10,
        help="Run validation every N training steps (default: %(default)s).",
    )

    return parser.parse_args()


def get_classification_df(args: argparse.Namespace) -> pd.DataFrame:
    # If the output file already exists, just load it
    if os.path.exists(args.classification_ds):
        logger.warning(
            "Found existing dataset at '%s'; loading it.", args.classification_ds
        )
        df: pd.DataFrame = pd.read_csv(args.classification_ds)
    else:
        logger.info(
            "No dataset found at '%s'; generating a new one.", args.classification_ds
        )
        df = create_classification_dataset(
            gb_file=args.gb_path,
            window_size=args.window_size,
            step_size=args.step_size,
            min_overlap_fraction=args.min_overlap_fraction,
        )
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(args.classification_ds), exist_ok=True)
        df.to_csv(args.classification_ds, index=False)
        logger.info("Saved newly created dataset to '%s'.", args.classification_ds)

    return df


def log_label_dist(df: pd.DataFrame) -> None:
    counts: pd.Series = df["label"].value_counts().sort_index()
    ratios: pd.Series = df["label"].value_counts(normalize=True).sort_index()
    dist_df: pd.DataFrame = pd.DataFrame({"count": counts, "ratio": ratios})
    logger.info("Label distribution:\n%s", dist_df)


def set_rand_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args: argparse.Namespace = parse_args()
    set_rand_seed(args.seed)

    wb_logger = None
    if args.wandb_logging:
        wandb.login()
        wb_logger = wandb.init(
            project="jepa-dna-pretrain",
            config=vars(args),
            dir="./logs",
        )

    experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if wb_logger:
        experiment_name = str(wb_logger.name)
    out_dir_ts = Path("data/experiments") / experiment_name
    out_dir_ts.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory for outputs: {out_dir_ts}")

    args_json = out_dir_ts / "args.json"
    with args_json.open("w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            f,
            indent=2,
        )

    df = get_classification_df(args)
    log_label_dist(df)
    logger.info(f"Classification dataframe shape = {df.shape}")

    if not args.run_classification:
        logger.info("Starting JEPA pretraining")
        train_jepa(
            classification_df=df,
            mask_prob=args.mask_prob,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            hidden_dim=args.hidden_dim,
            dim_feedforward=args.dim_feedforward,
            nhead=args.nhead,
            num_layers=args.num_layers,
            lr=args.lr,
            ema_tau=args.ema_tau,
            num_epochs=args.num_epochs,
            out_dir=out_dir_ts,
            wb_logger=wb_logger,
        )

    if args.run_classification:
        logger.info("Starting classification.")

        encoder_ckpt = out_dir_ts / "dna_encoder_jepa.pth"

        clf_on_raw = True
        if args.pretrained_enc:
            encoder_ckpt = Path(args.pretrained_enc)
            clf_on_raw = False
            logger.info(f"Will use pretrained encoder from {encoder_ckpt}")

        train_classifier(
            classification_df=df,
            encoder_ckpt=encoder_ckpt,
            hidden_dim=args.hidden_dim,
            dim_feedforward=args.dim_feedforward,
            nhead=args.nhead,
            num_layers=args.num_layers,
            clf_lr=args.clf_lr,
            clf_epochs=args.clf_epochs,
            clf_batch_size=args.clf_batch_size,
            val_steps=args.val_steps,
            out_dir=out_dir_ts,
            seed=args.seed,
            wb_logger=wb_logger,
            use_raw=clf_on_raw,
        )


if __name__ == "__main__":
    main()
