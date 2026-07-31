from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import CONFIG


class WeightedAverageRegressor(RegressorMixin, BaseEstimator):
    """Simple weighted ensemble compatible with sklearn and third-party estimators."""

    def __init__(self, estimators, weights=None) -> None:
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.estimators_ = []
        for name, estimator in self.estimators:
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        return self

    def predict(self, X):
        predictions = np.column_stack([
            estimator.predict(X) for _, estimator in self.estimators_
        ])
        if self.weights is None:
            return predictions.mean(axis=1)
        weights = np.asarray(self.weights, dtype=float)
        weights = weights / weights.sum()
        return np.average(predictions, axis=1, weights=weights)


class StackedRegressor(RegressorMixin, BaseEstimator):
    """Small custom stacking regressor.

    The base models are first trained on cross-validation folds to create
    out-of-fold predictions. A meta-model then learns how to combine those
    predictions. Finally, each base model is refit on the full training data.
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        cv: int = 5,
        random_state: int = 42,
        passthrough: bool = False,
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.random_state = random_state
        self.passthrough = passthrough

    @staticmethod
    def _take_rows(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    def _make_meta_features(self, X, base_predictions: np.ndarray) -> np.ndarray:
        if not self.passthrough:
            return base_predictions
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.column_stack([base_predictions, np.asarray(X)])

    def fit(self, X, y):
        y_array = np.asarray(y)
        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        oof_predictions = np.zeros((len(y_array), len(self.estimators)))
        self.estimators_ = []

        for model_index, (name, estimator) in enumerate(self.estimators):
            for train_idx, valid_idx in cv.split(X):
                fold_estimator = clone(estimator)
                fold_estimator.fit(
                    self._take_rows(X, train_idx),
                    y_array[train_idx],
                )
                oof_predictions[valid_idx, model_index] = fold_estimator.predict(
                    self._take_rows(X, valid_idx)
                )

            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X, y_array)
            self.estimators_.append((name, fitted_estimator))

        self.final_estimator_ = clone(
            self.final_estimator if self.final_estimator is not None else Ridge(alpha=1.0)
        )
        meta_X = self._make_meta_features(X, oof_predictions)
        self.final_estimator_.fit(meta_X, y_array)
        return self

    def predict(self, X):
        base_predictions = np.column_stack([
            estimator.predict(X) for _, estimator in self.estimators_
        ])
        meta_X = self._make_meta_features(X, base_predictions)
        return self.final_estimator_.predict(meta_X)


def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any] | None = None,
) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    config = CONFIG if config is None else config
    preprocessing = config["preprocessing"]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy=preprocessing["numeric_imputer_strategy"]),
        )
    ]
    if preprocessing.get("use_standard_scaler", True):
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing["categorical_imputer_strategy"],
                    fill_value=preprocessing["categorical_fill_value"],
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=preprocessing["one_hot_handle_unknown"],
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def create_lasso_model(config: dict[str, Any] | None = None) -> Lasso:
    """Create configured Lasso regressor."""
    config = CONFIG if config is None else config
    return Lasso(**config["models"]["lasso"])


def create_catboost_model(config: dict[str, Any] | None = None) -> CatBoostRegressor:
    """Create configured CatBoost regressor."""
    config = CONFIG if config is None else config
    return CatBoostRegressor(**config["models"]["catboost"])


def create_ensemble_model(config: dict[str, Any] | None = None) -> RegressorMixin:
    """Create either weighted-average ensemble or stacking ensemble."""
    config = CONFIG if config is None else config
    method = config["ensemble"].get("method", "weighted_average")
    base_estimators = [
        ("lasso", create_lasso_model(config=config)),
        ("catboost", create_catboost_model(config=config)),
    ]

    if method == "weighted_average":
        weights = config["ensemble"]["weights"]
        return WeightedAverageRegressor(
            estimators=base_estimators,
            weights=[weights["lasso"], weights["catboost"]],
        )

    if method == "stacking":
        stacking_cfg = config["models"].get("stacking", {})
        meta_model_name = stacking_cfg.get("meta_model", "ridge")
        if meta_model_name != "ridge":
            raise ValueError("Only meta_model='ridge' is currently supported.")

        return StackedRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=stacking_cfg.get("ridge_alpha", 1.0)),
            cv=stacking_cfg.get("cv", config["n_splits"]),
            random_state=config["random_state"],
            passthrough=stacking_cfg.get("passthrough", False),
        )

    raise ValueError(
        "Unknown ensemble method. Use 'weighted_average' or 'stacking'. "
        f"Got: {method}"
    )


def create_final_model(
    numeric_features: list[str],
    categorical_features: list[str],
    config: dict[str, Any] | None = None,
) -> Pipeline:
    """Create final model pipeline with preprocessing and configured ensemble."""
    config = CONFIG if config is None else config

    preprocessor = create_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", create_ensemble_model(config=config)),
        ]
    )


def create_cv(config: dict[str, Any] | None = None) -> KFold:
    """Create the configured cross-validation splitter."""
    config = CONFIG if config is None else config
    return KFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["random_state"],
    )


def regression_metrics_from_log_predictions(
    y_true_log: pd.Series | np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, float]:
    """Calculate Kaggle and business-friendly metrics from log-price predictions."""
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)

    y_true_price = np.expm1(y_true_log)
    y_pred_price = np.maximum(np.expm1(y_pred_log), 0)

    absolute_errors = np.abs(y_true_price - y_pred_price)

    return {
        "rmse_log": root_mean_squared_error(y_true_log, y_pred_log),
        "mae_price": mean_absolute_error(y_true_price, y_pred_price),
        "rmse_price": root_mean_squared_error(y_true_price, y_pred_price),
        "mape_percent": float(np.mean(absolute_errors / y_true_price) * 100),
        "wape_percent": float(absolute_errors.sum() / y_true_price.sum() * 100),
    }


def evaluate_model(
    model: RegressorMixin,
    X: pd.DataFrame,
    y_log: pd.Series,
    cv: KFold | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate a model with CV and out-of-fold business metrics.

    This function uses one explicit CV loop instead of calling both
    ``cross_val_score`` and ``cross_val_predict``. That avoids fitting the same
    expensive pipeline twice and keeps fold RMSE, OOF predictions, and business
    metrics based on the same validation folds.
    """
    config = CONFIG if config is None else config
    cv = create_cv(config=config) if cv is None else cv

    y_array = np.asarray(y_log)
    oof_predictions_log = np.zeros(len(y_array), dtype=float)
    fold_scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y_array):
        fold_model = clone(model)
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_valid = X.iloc[valid_idx] if hasattr(X, "iloc") else X[valid_idx]
        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]

        fold_model.fit(X_train, y_train)
        valid_predictions_log = fold_model.predict(X_valid)
        oof_predictions_log[valid_idx] = valid_predictions_log
        fold_scores.append(root_mean_squared_error(y_valid, valid_predictions_log))

    scores = np.asarray(fold_scores, dtype=float)

    metrics = {
        "cv_rmse_log_mean": float(scores.mean()),
        "cv_rmse_log_std": float(scores.std()),
    }
    oof_metrics = regression_metrics_from_log_predictions(
        y_true_log=y_array,
        y_pred_log=oof_predictions_log,
    )
    metrics.update({f"oof_{key}": value for key, value in oof_metrics.items()})

    return metrics
