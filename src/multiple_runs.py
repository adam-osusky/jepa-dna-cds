import argparse
import datetime
import logging
import re
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a main script 10 times, extract W&B run URLs from stdout, and save them to a specified file."
    )
    parser.add_argument(
        "--url_output_file",
        "-o",
        default="data/raw_wandb_urls.txt",
        help="Path to the file where extracted URLs will be appended",
    )
    parser.add_argument(
        "--pretrained_enc",
        default=None,
        # default="data/experiments/golden-vortex-56/dna_encoder_jepa.pth",
        help="Path to the pretrained encoder (default: %(default)s)."
        "If not specified will use classification head on new learnable embeddings",
    )
    args = parser.parse_args()

    # Name of the script to run 10 times
    script_name = "src.main"
    # File where all found URLs will be appended (from argparse)
    url_output_file = args.url_output_file

    # Regex pattern to find lines like:
    #   View run at https://wandb.ai/…
    # and capture the URL itself.
    pattern = re.compile(r"View run at (https://\S+)")
    pattern = re.compile(r"View run.*(https://wandb\.ai/\S+)")

    python_exe = sys.executable
    base_cmd = [python_exe, "-m", script_name, "--run_classification"]
    if args.pretrained_enc is not None:
        base_cmd += ["--pretrained_enc", args.pretrained_enc]

    for i in range(1, 11):
        seed_value = i
        logging.info(f"Starting run {i} of 10 with seed={seed_value}…")

        # Create a fresh cmd list for this iteration, inserting --seed
        cmd = base_cmd + ["--seed", str(seed_value)]
        logging.debug("Full command → %s", cmd)

        # Run the target script, capturing stdout & stderr
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)

        out = result.stdout or ""
        err = result.stderr or ""

        # Search stdout for the URL pattern
        match = pattern.search(out)
        if match:
            url = match.group(1)
            # Append the URL (with a timestamp) to the specified file
            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            with open(url_output_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | {i}-th run: {url}\n")

            logging.info(f"  Run {i}: Found URL → {url}")
            logging.info(f"      (appended to '{url_output_file}')")
        else:
            logging.info(f"  Run {i}: No matching URL found in stdout.")

        # Log if the script exited with a nonzero code
        if result.returncode != 0:
            logging.info(f"Script exited with code {result.returncode}.")
            logging.info(f"stderr:\n{err.strip()}")
        logging.info("-" * 60)


if __name__ == "__main__":
    main()
