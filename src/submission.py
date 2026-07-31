from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from configs.config import CONFIG


class PredictsPricesLog(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


def create_submission(
    model: PredictsPricesLog,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create Kaggle submission dataframe from fitted log-price model."""
    config = CONFIG if config is None else config
    id_column = config["columns"]["id"]
    target_column = config["columns"]["target"]

    predictions_log = model.predict(test_features)
    predictions = np.maximum(np.expm1(predictions_log), 0)

    return pd.DataFrame(
        {
            id_column: test_ids,
            target_column: predictions,
        }
    )


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    """Validate submission format before saving."""
    config = CONFIG if config is None else config
    id_column = config["columns"]["id"]
    target_column = config["columns"]["target"]

    if submission.shape != sample_submission.shape:
        raise ValueError(
            f"Invalid submission shape: {submission.shape}. "
            f"Expected: {sample_submission.shape}."
        )

    if submission[target_column].isna().sum() > 0:
        raise ValueError("Submission contains missing predictions.")

    if (submission[target_column] < 0).sum() > 0:
        raise ValueError("Submission contains negative predictions.")

    if not submission[id_column].equals(sample_submission[id_column]):
        raise ValueError("Submission IDs do not match sample submission IDs.")
