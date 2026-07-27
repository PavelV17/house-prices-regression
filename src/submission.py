import numpy as np
import pandas as pd

from configs.config import ID_COLUMN, TARGET_COLUMN


def create_submission(
    model,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
) -> pd.DataFrame:
    """Create Kaggle submission dataframe from fitted model and test features."""
    predictions_log = model.predict(test_features)
    predictions = np.expm1(predictions_log)

    predictions = np.maximum(predictions, 0)

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            TARGET_COLUMN: predictions,
        }
    )

    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate submission format before saving."""
    if submission.shape != sample_submission.shape:
        raise ValueError(
            f"Invalid submission shape: {submission.shape}. "
            f"Expected: {sample_submission.shape}."
        )

    if submission[TARGET_COLUMN].isna().sum() > 0:
        raise ValueError("Submission contains missing predictions.")

    if (submission[TARGET_COLUMN] < 0).sum() > 0:
        raise ValueError("Submission contains negative predictions.")

    if not submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise ValueError("Submission IDs do not match sample submission IDs.")