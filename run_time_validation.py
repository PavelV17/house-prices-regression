from __future__ import annotations

import argparse

from configs.config import load_config
from src.time_validation import run_time_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production-style time-based validation for House Prices."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_results, summary = run_time_validation(config=config, save_artifacts=True)

    print("Time-based validation finished.")
    print(f"Splits evaluated: {len(split_results)}")
    print("\nTest metric summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
