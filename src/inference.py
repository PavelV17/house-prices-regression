from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from configs.config import CONFIG, get_artifact_paths
from src.data import load_data
from src.features import add_features, apply_raw_data_fixes
from src.submission import create_submission, validate_submission


def load_model(
    config: dict[str, Any] | None = None,
    model_path=None,
) -> Pipeline:
    """Load a fitted model from disk."""
    config = CONFIG if config is None else config
    path = get_artifact_paths(config)["final_model"] if model_path is None else model_path
    return joblib.load(path)


def create_submission_from_model(
    model: Pipeline,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
    sample_submission: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create and validate a Kaggle submission from a fitted model."""
    config = CONFIG if config is None else config
    submission = create_submission(
        model=model,
        test_features=test_features,
        test_ids=test_ids,
        config=config,
    )
    validate_submission(
        submission=submission,
        sample_submission=sample_submission,
        config=config,
    )
    return submission


def run_inference(
    config: dict[str, Any] | None = None,
    model_path=None,
    submission_path=None,
) -> pd.DataFrame:
    """Load a saved model and create a submission for the Kaggle test set."""
    config = CONFIG if config is None else config
    artifact_paths = get_artifact_paths(config)
    submission_path = artifact_paths["final_submission"] if submission_path is None else submission_path
    id_column = config["columns"]["id"]

    _, test, sample_submission = load_data(config=config)
    test = apply_raw_data_fixes(test, config=config)
    test_features = add_features(test)
    model = load_model(config=config, model_path=model_path)

    submission = create_submission_from_model(
        model=model,
        test_features=test_features,
        test_ids=test[id_column],
        sample_submission=sample_submission,
        config=config,
    )

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    return submission
