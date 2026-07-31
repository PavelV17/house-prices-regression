from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

import optuna
import pandas as pd

from configs.config import get_artifact_paths, load_config, save_config
from src.data import load_data
from src.features import (
    add_features,
    apply_raw_data_fixes,
    get_feature_types,
    prepare_features_and_target,
    remove_outliers,
)
from src.models import create_final_model, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune House Prices model hyperparameters with Optuna."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config relative to project root or absolute path.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials. Overrides optuna.n_trials from config.",
    )
    parser.add_argument(
        "--experiment",
        default="optuna_tuning",
        help="Study name for logging.",
    )
    return parser.parse_args()


def _float_space(
    search_space: dict[str, Any],
    name: str,
    default_low: float,
    default_high: float,
    default_log: bool = False,
) -> tuple[float, float, bool]:
    """Read a float search-space rule from config with safe defaults."""
    rule = search_space.get(name, {})
    return (
        float(rule.get("low", default_low)),
        float(rule.get("high", default_high)),
        bool(rule.get("log", default_log)),
    )


def _int_space(
    search_space: dict[str, Any],
    name: str,
    default_low: int,
    default_high: int,
    default_step: int = 1,
) -> tuple[int, int, int]:
    """Read an integer search-space rule from config with safe defaults."""
    rule = search_space.get(name, {})
    return (
        int(rule.get("low", default_low)),
        int(rule.get("high", default_high)),
        int(rule.get("step", default_step)),
    )


def build_trial_config(base_config: dict[str, Any], trial: optuna.Trial) -> dict[str, Any]:
    """Create a config variant suggested by Optuna.

    The default search space is intentionally focused around the already strong
    Lasso + CatBoost solution. This avoids wasting long runs on obviously weak
    regions, while keeping the ranges configurable from YAML.
    """
    config = deepcopy(base_config)
    search_space = base_config.get("optuna", {}).get("search_space", {})

    low, high, log = _float_space(
        search_space, "lasso_alpha", 1e-4, 2e-3, default_log=True
    )
    config["models"]["lasso"]["alpha"] = trial.suggest_float(
        "lasso_alpha", low, high, log=log
    )

    low_i, high_i, step_i = _int_space(
        search_space, "catboost_iterations", 500, 1000, default_step=100
    )
    config["models"]["catboost"]["iterations"] = trial.suggest_int(
        "catboost_iterations", low_i, high_i, step=step_i
    )

    low, high, log = _float_space(
        search_space, "catboost_learning_rate", 0.02, 0.07, default_log=True
    )
    config["models"]["catboost"]["learning_rate"] = trial.suggest_float(
        "catboost_learning_rate", low, high, log=log
    )

    low_i, high_i, step_i = _int_space(
        search_space, "catboost_depth", 4, 6, default_step=1
    )
    config["models"]["catboost"]["depth"] = trial.suggest_int(
        "catboost_depth", low_i, high_i, step=step_i
    )

    low, high, log = _float_space(
        search_space, "catboost_l2_leaf_reg", 1.0, 5.0, default_log=True
    )
    config["models"]["catboost"]["l2_leaf_reg"] = trial.suggest_float(
        "catboost_l2_leaf_reg", low, high, log=log
    )

    low, high, log = _float_space(
        search_space, "ensemble_lasso_weight", 0.45, 0.75, default_log=False
    )
    lasso_weight = trial.suggest_float("ensemble_lasso_weight", low, high, log=log)
    config["ensemble"]["method"] = "weighted_average"
    config["ensemble"]["weights"] = {
        "lasso": lasso_weight,
        "catboost": 1.0 - lasso_weight,
    }
    return config


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    artifact_paths = get_artifact_paths(base_config)

    train, _, _ = load_data(config=base_config)
    train = apply_raw_data_fixes(train, config=base_config)
    train = remove_outliers(train, config=base_config)
    train = add_features(train)

    X, y_log = prepare_features_and_target(train, config=base_config)
    numeric_features, categorical_features = get_feature_types(X, config=base_config)

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_config = build_trial_config(base_config, trial)
        model = create_final_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            config=trial_config,
        )
        metrics = evaluate_model(model, X, y_log, config=trial_config)
        trial.set_user_attr("metrics", metrics)
        trial_rows.append({
            "trial": trial.number,
            "value": metrics["cv_rmse_log_mean"],
            **trial.params,
            **metrics,
        })
        return metrics["cv_rmse_log_mean"]

    sampler = optuna.samplers.TPESampler(seed=base_config["random_state"])
    study = optuna.create_study(
        direction="minimize",
        study_name=args.experiment,
        sampler=sampler,
    )
    n_trials = args.trials if args.trials is not None else base_config["optuna"]["n_trials"]
    timeout = base_config["optuna"].get("timeout")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    artifact_paths["results_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(artifact_paths["optuna_trials"], index=False)

    best_config = build_trial_config(base_config, study.best_trial)
    save_config(best_config, artifact_paths["optuna_best_config"])

    print("Optuna tuning completed.")
    print(f"Best CV RMSE log: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved trials to: {artifact_paths['optuna_trials']}")
    print(f"Saved best config to: {artifact_paths['optuna_best_config']}")
    print("Run the best config with:")
    print(f"python main.py --config {artifact_paths['optuna_best_config']} --experiment optuna_best")


if __name__ == "__main__":
    main()
