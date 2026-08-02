from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.config import CONFIG, get_artifact_paths, get_data_paths
from src.features import (
    add_features,
    apply_raw_data_fixes,
    get_feature_types,
    prepare_features_and_target,
    remove_outliers,
)
from src.models import create_final_model, regression_metrics_from_log_predictions


@dataclass(frozen=True)
class TimeSplit:
    """One expanding-window train/validation/test split."""

    split: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def make_sale_period(data: pd.DataFrame) -> pd.Series:
    """Create a sortable monthly period from YrSold and MoSold."""
    return (
        pd.to_numeric(data["YrSold"], errors="raise").astype(int) * 12
        + pd.to_numeric(data["MoSold"], errors="raise").astype(int)
    )


def format_period(period: int) -> str:
    """Convert integer month period back to YYYY-MM."""
    year = (period - 1) // 12
    month = period - year * 12
    return f"{year:04d}-{month:02d}"


def format_period_range(start: int, end: int) -> str:
    """Format period range as YYYY-MM..YYYY-MM."""
    return f"{format_period(start)}..{format_period(end)}"


def generate_expanding_time_splits(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> list[TimeSplit]:
    """Generate expanding-window splits ordered by sale month.

    The training window expands over time. Validation and test windows are
    future periods relative to the training window, which better imitates a
    production setting than shuffled KFold for dated transactions.
    """
    config = CONFIG if config is None else config
    validation_cfg = config["time_validation"]

    min_train_months = int(validation_cfg["min_train_months"])
    validation_months = int(validation_cfg["validation_months"])
    test_months = int(validation_cfg["test_months"])
    step_months = int(validation_cfg["step_months"])
    min_train_size = int(validation_cfg["min_train_size"])
    min_validation_size = int(validation_cfg["min_validation_size"])
    min_test_size = int(validation_cfg["min_test_size"])

    periods = make_sale_period(data)
    first_period = int(periods.min())
    last_period = int(periods.max())

    first_train_end = first_period + min_train_months - 1
    last_train_end = last_period - validation_months - test_months

    splits: list[TimeSplit] = []
    split_number = 1

    train_end = first_train_end
    while train_end <= last_train_end:
        val_start = train_end + 1
        val_end = train_end + validation_months
        test_start = val_end + 1
        test_end = val_end + test_months

        train_mask = periods <= train_end
        validation_mask = (periods >= val_start) & (periods <= val_end)
        test_mask = (periods >= test_start) & (periods <= test_end)

        if (
            int(train_mask.sum()) >= min_train_size
            and int(validation_mask.sum()) >= min_validation_size
            and int(test_mask.sum()) >= min_test_size
        ):
            splits.append(
                TimeSplit(
                    split=split_number,
                    train_start=first_period,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            split_number += 1

        train_end += step_months

    if not splits:
        raise ValueError(
            "No time-validation splits were generated. Try lowering "
            "min_train_size/min_validation_size/min_test_size or shortening "
            "validation_months/test_months."
        )

    return splits


def evaluate_fitted_model_on_holdout(
    model,
    X: pd.DataFrame,
    y_log: pd.Series,
    prefix: str,
) -> dict[str, float]:
    """Predict one holdout set and return prefixed regression metrics."""
    predictions_log = model.predict(X)
    metrics = regression_metrics_from_log_predictions(
        y_true_log=y_log,
        y_pred_log=predictions_log,
    )
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def run_time_validation(
    config: dict[str, Any] | None = None,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run production-style time-based validation.

    Only train.csv is used, because it contains SalePrice. For every split:
    - the model is fitted only on past sales;
    - validation is the next period;
    - test is a still later future period;
    - preprocessing is fitted inside the model pipeline only on the training part.

    The test period is not used for model selection. Its metrics are aggregated
    into mean/std/min/max statistics.
    """
    config = CONFIG if config is None else config
    target_column = config["columns"]["target"]

    train_path = get_data_paths(config)["train"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Put train.csv into data/ first."
        )

    raw_data = pd.read_csv(train_path)
    raw_data = apply_raw_data_fixes(raw_data, config=config)
    raw_data = raw_data.assign(_sale_period=make_sale_period(raw_data))

    splits = generate_expanding_time_splits(raw_data, config=config)
    rows: list[dict[str, Any]] = []

    for split in splits:
        train_mask = raw_data["_sale_period"] <= split.train_end
        validation_mask = (
            (raw_data["_sale_period"] >= split.val_start)
            & (raw_data["_sale_period"] <= split.val_end)
        )
        test_mask = (
            (raw_data["_sale_period"] >= split.test_start)
            & (raw_data["_sale_period"] <= split.test_end)
        )

        train_part = raw_data.loc[train_mask].drop(columns=["_sale_period"]).copy()
        validation_part = raw_data.loc[validation_mask].drop(columns=["_sale_period"]).copy()
        test_part = raw_data.loc[test_mask].drop(columns=["_sale_period"]).copy()

        # Known training outliers are removed from the training window only.
        # Validation/test represent future production-like data and are evaluated untouched.
        train_part = remove_outliers(train_part, config=config)

        train_features = add_features(train_part)
        validation_features = add_features(validation_part)
        test_features = add_features(test_part)

        X_train, y_train_log = prepare_features_and_target(train_features, config=config)
        X_validation, y_validation_log = prepare_features_and_target(
            validation_features,
            config=config,
        )
        X_test, y_test_log = prepare_features_and_target(test_features, config=config)

        numeric_features, categorical_features = get_feature_types(X_train, config=config)
        model = create_final_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=config,
        )
        model.fit(X_train, y_train_log)

        row: dict[str, Any] = {
            "split": split.split,
            "train_period": format_period_range(split.train_start, split.train_end),
            "validation_period": format_period_range(split.val_start, split.val_end),
            "test_period": format_period_range(split.test_start, split.test_end),
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_validation)),
            "test_size": int(len(X_test)),
            "target_mean_train": float(train_part[target_column].mean()),
            "target_mean_validation": float(validation_part[target_column].mean()),
            "target_mean_test": float(test_part[target_column].mean()),
        }
        row.update(
            evaluate_fitted_model_on_holdout(
                model=model,
                X=X_validation,
                y_log=y_validation_log,
                prefix="validation",
            )
        )
        row.update(
            evaluate_fitted_model_on_holdout(
                model=model,
                X=X_test,
                y_log=y_test_log,
                prefix="test",
            )
        )
        rows.append(row)

    split_results = pd.DataFrame(rows)

    metric_names = [
        "rmse_log",
        "mae_price",
        "rmse_price",
        "mape_percent",
        "wape_percent",
    ]
    test_metric_columns = [f"test_{metric_name}" for metric_name in metric_names]
    summary = (
        split_results[test_metric_columns]
        .agg(["mean", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    summary.insert(1, "n_splits", len(split_results))

    if save_artifacts:
        artifact_paths = get_artifact_paths(config)
        results_dir = artifact_paths["results_dir"]
        results_dir.mkdir(parents=True, exist_ok=True)
        split_results.to_csv(artifact_paths["time_validation_splits"], index=False)
        summary.to_csv(artifact_paths["time_validation_summary"], index=False)

    return split_results, summary
