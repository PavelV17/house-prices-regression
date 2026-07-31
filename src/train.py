from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from configs.config import CONFIG, get_artifact_paths
from src.data import load_data
from src.features import (
    add_features,
    apply_raw_data_fixes,
    get_feature_types,
    prepare_features_and_target,
    remove_outliers,
)
from src.models import create_final_model, evaluate_model


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, float]
    test_features: pd.DataFrame
    test_ids: pd.Series
    sample_submission: pd.DataFrame


def save_metrics(
    metrics: dict[str, float],
    config: dict[str, Any] | None = None,
) -> None:
    """Save final model metrics to CSV."""
    config = CONFIG if config is None else config
    path = get_artifact_paths(config)["final_metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def save_model(
    model: Pipeline,
    config: dict[str, Any] | None = None,
) -> None:
    """Save fitted model to disk."""
    config = CONFIG if config is None else config
    path = get_artifact_paths(config)["final_model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def build_experiment_row(
    metrics: dict[str, float],
    config: dict[str, Any],
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Flatten key config values and metrics into one experiment-tracking row."""
    name = experiment_name or config.get("experiment", {}).get("name", "experiment")
    lasso = config["models"]["lasso"]
    catboost = config["models"]["catboost"]
    weights = config["ensemble"]["weights"]
    preprocessing = config["preprocessing"]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": name,
        "n_splits": config["n_splits"],
        "remove_outliers": preprocessing.get("remove_outliers", True),
        "outlier_ids": ",".join(map(str, preprocessing.get("outlier_ids", []))),
        "categorical_numeric_features": ",".join(
            preprocessing.get("categorical_numeric_features", [])
        ),
        "ensemble_method": config["ensemble"].get("method", "weighted_average"),
        "lasso_alpha": lasso.get("alpha"),
        "lasso_max_iter": lasso.get("max_iter"),
        "catboost_iterations": catboost.get("iterations"),
        "catboost_learning_rate": catboost.get("learning_rate"),
        "catboost_depth": catboost.get("depth"),
        "catboost_l2_leaf_reg": catboost.get("l2_leaf_reg"),
        "ensemble_lasso_weight": weights.get("lasso"),
        "ensemble_catboost_weight": weights.get("catboost"),
        **metrics,
    }


def append_experiment_result(
    metrics: dict[str, float],
    config: dict[str, Any],
    experiment_name: str | None = None,
) -> None:
    """Append one run to results/experiments.csv."""
    path = get_artifact_paths(config)["experiments"]
    path.parent.mkdir(parents=True, exist_ok=True)

    row = build_experiment_row(
        metrics=metrics,
        config=config,
        experiment_name=experiment_name,
    )
    new_result = pd.DataFrame([row])

    if path.exists():
        previous = pd.read_csv(path)
        results = pd.concat([previous, new_result], ignore_index=True)
    else:
        results = new_result

    results.to_csv(path, index=False)


def train_final_model(
    save_artifacts: bool = True,
    config: dict[str, Any] | None = None,
    experiment_name: str | None = None,
) -> TrainingResult:
    """Train the final House Prices model and return all artifacts for inference."""
    config = CONFIG if config is None else config
    id_column = config["columns"]["id"]

    train, test, sample_submission = load_data(config=config)

    train = apply_raw_data_fixes(train, config=config)
    test = apply_raw_data_fixes(test, config=config)

    train_clean = remove_outliers(train, config=config)
    train_fe = add_features(train_clean)
    test_fe = add_features(test)

    X, y_log = prepare_features_and_target(train_fe, config=config)
    numeric_features, categorical_features = get_feature_types(X, config=config)

    final_model = create_final_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    metrics = evaluate_model(final_model, X, y_log, config=config)
    final_model.fit(X, y_log)

    if save_artifacts:
        artifact_paths = get_artifact_paths(config)
        artifact_paths["results_dir"].mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, config=config)
        save_model(final_model, config=config)
        append_experiment_result(
            metrics=metrics,
            config=config,
            experiment_name=experiment_name,
        )

    return TrainingResult(
        model=final_model,
        metrics=metrics,
        test_features=test_fe,
        test_ids=test[id_column],
        sample_submission=sample_submission,
    )
