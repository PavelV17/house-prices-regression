from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively update base dictionary with override values."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "baseline"},
    "random_state": 42,
    "n_splits": 5,
    "scoring": "neg_root_mean_squared_error",
    "paths": {
        "data_dir": "data",
        "results_dir": "results",
        "submissions_dir": "submissions",
        "train_file": "train.csv",
        "test_file": "test.csv",
        "sample_submission_file": "sample_submission.csv",
        "data_description_file": "data_description.txt",
    },
    "columns": {"id": "Id", "target": "SalePrice"},
    "preprocessing": {
        "remove_outliers": True,
        "outlier_ids": [524, 1299],
        "categorical_numeric_features": ["MSSubClass"],
        "invalid_numeric_values": {
            "GarageYrBlt": {
                "min": 1800,
                "max_relative_to_column": "YrSold",
                "replacement": None,
            }
        },
        "numeric_imputer_strategy": "median",
        "categorical_imputer_strategy": "constant",
        "categorical_fill_value": "None",
        "one_hot_handle_unknown": "ignore",
        "use_standard_scaler": True,
    },
    "models": {
        "lasso": {"alpha": 0.0005, "max_iter": 20000, "random_state": 42},
        "catboost": {
            "iterations": 800,
            "learning_rate": 0.03,
            "depth": 5,
            "l2_leaf_reg": 3,
            "loss_function": "RMSE",
            "random_state": 42,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": -1,
        },
        "stacking": {
            "meta_model": "ridge",
            "ridge_alpha": 1.0,
            "cv": 5,
            "passthrough": False,
        },
    },
    "ensemble": {
        "method": "weighted_average",
        "weights": {"lasso": 0.6, "catboost": 0.4},
    },
    "torch_mlp": {
        "validation_size": 0.2,
        "hidden_dims": [256, 128],
        "dropout": 0.10,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "epochs": 250,
        "patience": 25,
        "device": "auto",
    },
    "optuna": {
        "n_trials": 20,
        "timeout": None,
        "output_filename": "optuna_trials.csv",
        "best_config_filename": "optuna_best_config.yaml",
        "search_space": {
            "lasso_alpha": {"low": 0.0001, "high": 0.002, "log": True},
            "catboost_iterations": {"low": 500, "high": 1000, "step": 100},
            "catboost_learning_rate": {"low": 0.02, "high": 0.07, "log": True},
            "catboost_depth": {"low": 4, "high": 6, "step": 1},
            "catboost_l2_leaf_reg": {"low": 1.0, "high": 5.0, "log": True},
            "ensemble_lasso_weight": {"low": 0.45, "high": 0.75, "log": False},
        },
    },
    "time_validation": {
        "min_train_months": 24,
        "validation_months": 6,
        "test_months": 6,
        "step_months": 6,
        "min_train_size": 400,
        "min_validation_size": 50,
        "min_test_size": 50,
    },
    "artifacts": {
        "final_model_filename": "final_lasso_catboost_ensemble.joblib",
        "final_metrics_filename": "final_model_metrics.csv",
        "all_model_results_filename": "all_model_results.csv",
        "experiments_filename": "experiments.csv",
        "final_submission_filename": "final_lasso_catboost_ensemble_submission.csv",
        "torch_mlp_metrics_filename": "torch_mlp_metrics.csv",
        "optuna_trials_filename": "optuna_trials.csv",
        "optuna_best_config_filename": "optuna_best_config.yaml",
        "time_validation_splits_filename": "time_validation_splits.csv",
        "time_validation_summary_filename": "time_validation_summary.csv",
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge it with safe defaults."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    return _deep_update(DEFAULT_CONFIG, loaded)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config as YAML."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def project_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_data_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Return resolved paths to Kaggle data files."""
    paths = config["paths"]
    data_dir = project_path(paths["data_dir"])
    return {
        "data_dir": data_dir,
        "train": data_dir / paths["train_file"],
        "test": data_dir / paths["test_file"],
        "sample_submission": data_dir / paths["sample_submission_file"],
        "data_description": data_dir / paths["data_description_file"],
    }


def get_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Return resolved output artifact paths."""
    paths = config["paths"]
    artifacts = config["artifacts"]
    results_dir = project_path(paths["results_dir"])
    submissions_dir = project_path(paths["submissions_dir"])
    return {
        "results_dir": results_dir,
        "submissions_dir": submissions_dir,
        "final_model": results_dir / artifacts["final_model_filename"],
        "final_metrics": results_dir / artifacts["final_metrics_filename"],
        "all_model_results": results_dir / artifacts["all_model_results_filename"],
        "experiments": results_dir / artifacts["experiments_filename"],
        "final_submission": submissions_dir / artifacts["final_submission_filename"],
        "torch_mlp_metrics": results_dir / artifacts["torch_mlp_metrics_filename"],
        "optuna_trials": results_dir / artifacts["optuna_trials_filename"],
        "optuna_best_config": results_dir / artifacts["optuna_best_config_filename"],
        "time_validation_splits": results_dir / artifacts["time_validation_splits_filename"],
        "time_validation_summary": results_dir / artifacts["time_validation_summary_filename"],
    }


# Backward-compatible constants for notebooks or older imports.
CONFIG = load_config()
DATA_PATHS = get_data_paths(CONFIG)
ARTIFACT_PATHS = get_artifact_paths(CONFIG)

DATA_DIR = DATA_PATHS["data_dir"]
RESULTS_DIR = ARTIFACT_PATHS["results_dir"]
SUBMISSIONS_DIR = ARTIFACT_PATHS["submissions_dir"]

TRAIN_PATH = DATA_PATHS["train"]
TEST_PATH = DATA_PATHS["test"]
SAMPLE_SUBMISSION_PATH = DATA_PATHS["sample_submission"]
DATA_DESCRIPTION_PATH = DATA_PATHS["data_description"]

TARGET_COLUMN = CONFIG["columns"]["target"]
ID_COLUMN = CONFIG["columns"]["id"]

RANDOM_STATE = CONFIG["random_state"]
N_SPLITS = CONFIG["n_splits"]
SCORING = CONFIG["scoring"]
OUTLIER_IDS = CONFIG["preprocessing"]["outlier_ids"]
CATEGORICAL_NUMERIC_FEATURES = CONFIG["preprocessing"]["categorical_numeric_features"]
INVALID_NUMERIC_VALUES = CONFIG["preprocessing"]["invalid_numeric_values"]

NUMERIC_IMPUTER_STRATEGY = CONFIG["preprocessing"]["numeric_imputer_strategy"]
CATEGORICAL_IMPUTER_STRATEGY = CONFIG["preprocessing"]["categorical_imputer_strategy"]
CATEGORICAL_FILL_VALUE = CONFIG["preprocessing"]["categorical_fill_value"]
ONE_HOT_HANDLE_UNKNOWN = CONFIG["preprocessing"]["one_hot_handle_unknown"]

LASSO_PARAMS = CONFIG["models"]["lasso"]
CATBOOST_PARAMS = CONFIG["models"]["catboost"]
ENSEMBLE_METHOD = CONFIG["ensemble"]["method"]
FINAL_ENSEMBLE_WEIGHTS = [
    CONFIG["ensemble"]["weights"]["lasso"],
    CONFIG["ensemble"]["weights"]["catboost"],
]

FINAL_MODEL_PATH = ARTIFACT_PATHS["final_model"]
FINAL_METRICS_PATH = ARTIFACT_PATHS["final_metrics"]
ALL_MODEL_RESULTS_PATH = ARTIFACT_PATHS["all_model_results"]
EXPERIMENTS_PATH = ARTIFACT_PATHS["experiments"]
FINAL_SUBMISSION_PATH = ARTIFACT_PATHS["final_submission"]
TORCH_MLP_METRICS_PATH = ARTIFACT_PATHS["torch_mlp_metrics"]
OPTUNA_TRIALS_PATH = ARTIFACT_PATHS["optuna_trials"]
OPTUNA_BEST_CONFIG_PATH = ARTIFACT_PATHS["optuna_best_config"]
TIME_VALIDATION_SPLITS_PATH = ARTIFACT_PATHS["time_validation_splits"]
TIME_VALIDATION_SUMMARY_PATH = ARTIFACT_PATHS["time_validation_summary"]
