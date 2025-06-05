import argparse
import logging
import os
import re
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def strip_ansi_codes(text: str) -> str:
    """
    Remove ANSI escape codes (e.g., '\x1b[0m') from a string.
    """
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def extract_wandb_urls(line: str) -> list:
    """
    Given a line of text, return a list of WandB URLs found in that line.
    """
    # Strip any ANSI codes first
    clean_line = strip_ansi_codes(line)

    # Regex to match URLs like: https://wandb.ai/entity/project/runs/run_id
    url_pattern = re.compile(r"https://wandb\.ai/[^\s]+")
    return url_pattern.findall(clean_line)


def parse_run_path(url: str) -> str:
    """
    Given a WandB run URL, parse and return the "entity/project/run_id" string.
    """
    parsed = urlparse(url)
    # parsed.path -> "/gallus/jepa-dna-pretrain/runs/9jx6crvo"
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 4 and parts[2] == "runs":
        entity = parts[0]
        project = parts[1]
        run_id = parts[3]
        return f"{entity}/{project}/{run_id}"
    else:
        raise ValueError(f"URL does not match expected pattern: {url}")


def load_runs_from_file(filepath: str):
    """
    Read the given text file line by line, extract WandB URLs, and load
    each run via the WandB API.
    """
    api = wandb.Api()
    runs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            urls = extract_wandb_urls(line)
            for url in urls:
                try:
                    run_path = parse_run_path(url)
                    run = api.run(run_path)
                    runs.append(run)
                    logging.info(f"[Line {lineno}] Loaded run: {run_path}")
                except Exception as e:
                    logging.info(f"[Line {lineno}] Skipped URL '{url}': {e}")
    return runs


def compute_epoch_summary(runs: list) -> pd.DataFrame:
    """
    Given a list of wandb.Run objects, fetch each run's epoch-level train/val accuracy,
    and compute a DataFrame with columns ['epoch', 'train_mean', 'train_std',
    'val_mean', 'val_std'].
    Assumes all runs have the same number of epochs.
    """
    dfs_per_run = []

    for run in runs:
        # Fetch full history (raise samples high enough to avoid truncation)
        hist = run.history(keys=None, samples=20_000)

        # Keep only rows where epoch-level metrics are not NaN
        epoch_rows = hist.dropna(
            subset=["clf_raw_epoch_train_acc", "clf_raw_epoch_val_acc"]
        )

        # Reset index so we can assign epoch = 1, 2, 3, ... sequentially
        epoch_rows = epoch_rows.reset_index(drop=True)
        epoch_rows["epoch"] = epoch_rows.index + 1

        # Extract just the epoch, train_acc, and val_acc columns
        df_small = epoch_rows[
            ["epoch", "clf_raw_epoch_train_acc", "clf_raw_epoch_val_acc"]
        ].copy()
        df_small.columns = ["epoch", "train_acc", "val_acc"]

        # Tag with run_id (for debugging if needed)
        df_small["run_id"] = run.id

        dfs_per_run.append(df_small)

    # Concatenate all runs into one DataFrame
    all_epochs = pd.concat(dfs_per_run, axis=0, ignore_index=True)

    # Group by epoch and compute mean and std for train_acc and val_acc
    grouped = all_epochs.groupby("epoch")
    summary = grouped.agg({"train_acc": ["mean", "std"], "val_acc": ["mean", "std"]})

    # Flatten MultiIndex columns
    summary.columns = ["train_mean", "train_std", "val_mean", "val_std"]
    summary = summary.reset_index()

    return summary


def plot_and_save(summary_df: pd.DataFrame, output_path: str):
    """
    Given a summary DataFrame with ['epoch', 'train_mean', 'train_std',
    'val_mean', 'val_std'], plot the mean ±1σ for train/val accuracy and save to output_path.
    """
    plt.figure(figsize=(16, 10))

    # Plot train accuracy mean
    plt.plot(
        summary_df["epoch"],
        summary_df["train_mean"],
        label="Train Accuracy (mean)",
        linestyle="-",
        marker="o",
    )
    # Shade ±1σ around train mean
    plt.fill_between(
        summary_df["epoch"],
        summary_df["train_mean"] - summary_df["train_std"],
        summary_df["train_mean"] + summary_df["train_std"],
        alpha=0.3,
    )

    # Plot val accuracy mean
    plt.plot(
        summary_df["epoch"],
        summary_df["val_mean"],
        label="Val Accuracy (mean)",
        linestyle="-",
        marker="s",
    )
    # Shade ±1σ around val mean
    plt.fill_between(
        summary_df["epoch"],
        summary_df["val_mean"] - summary_df["val_std"],
        summary_df["val_mean"] + summary_df["val_std"],
        alpha=0.3,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Mean ±1 σ of Train/Val Accuracy Across Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load WandB runs from a text file containing WandB URLs."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to the text file with WandB URLs (one or more per line).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="epoch_accuracy_summary.png",
        help="Path (including filename) where the resulting plot will be saved. Default: epoch_accuracy_summary.png",
    )
    args = parser.parse_args()

    runs = load_runs_from_file(args.filepath)

    logging.info(f"Total runs loaded: {len(runs)}")

    if not runs:
        print("No runs were loaded. Exiting.")
        return

    summary_df = compute_epoch_summary(runs)
    plot_and_save(summary_df, args.output)


if __name__ == "__main__":
    main()
