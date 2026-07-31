from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import CONFIG, get_data_paths


def validate_data_files(
    paths: list[Path] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Check that all required Kaggle data files are available locally."""
    config = CONFIG if config is None else config
    data_paths = get_data_paths(config)
    required_paths = [
        data_paths["train"],
        data_paths["test"],
        data_paths["sample_submission"],
    ]
    paths = required_paths if paths is None else paths
    missing_paths = [path for path in paths if not path.exists()]

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required data files were not found:\n"
            f"{missing}\n\n"
            "Download the House Prices data from Kaggle and put the CSV files "
            "into the project data/ directory. See README.md for details."
        )


def load_data(
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load House Prices train, test and sample submission datasets."""
    config = CONFIG if config is None else config
    validate_data_files(config=config)
    data_paths = get_data_paths(config)

    train = pd.read_csv(data_paths["train"])
    test = pd.read_csv(data_paths["test"])
    sample_submission = pd.read_csv(data_paths["sample_submission"])

    return train, test, sample_submission
