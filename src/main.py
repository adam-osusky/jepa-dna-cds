import argparse
import logging
import os

import pandas as pd

from src.window import create_classification_dataset

# Configure logging
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

    return parser.parse_args()


def get_classification_df(args: argparse.Namespace) -> pd.DataFrame:
    # If the output file already exists, just load it
    if os.path.exists(args.classification_ds):
        logger.info(
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


def main() -> None:
    args: argparse.Namespace = parse_args()

    df = get_classification_df(args)
    log_label_dist(df)


if __name__ == "__main__":
    main()
