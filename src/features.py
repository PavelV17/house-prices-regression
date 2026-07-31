from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from configs.config import CONFIG


def _category_code_to_string(value: object) -> object:
    """Convert category-like numeric codes to clean strings while preserving missing values."""
    if pd.isna(value):
        return np.nan
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return str(value)


def apply_raw_data_fixes(
    data: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply safe raw-data fixes before feature engineering.

    Current fixes:
    - treat numeric-looking category codes such as MSSubClass as categorical;
    - replace impossible numeric values such as GarageYrBlt=2207 or GarageYrBlt > YrSold with missing.
    """
    config = CONFIG if config is None else config
    data = data.copy()
    preprocessing = config["preprocessing"]

    for column in preprocessing.get("categorical_numeric_features", []):
        if column in data.columns:
            data[column] = data[column].map(_category_code_to_string).astype("object")

    for column, rule in preprocessing.get("invalid_numeric_values", {}).items():
        if column not in data.columns:
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        mask = pd.Series(False, index=data.index)

        if rule.get("min") is not None:
            mask |= values < rule["min"]

        if rule.get("max") is not None:
            mask |= values > rule["max"]

        max_relative_to_column = rule.get("max_relative_to_column")
        if max_relative_to_column is not None and max_relative_to_column in data.columns:
            relative_values = pd.to_numeric(
                data[max_relative_to_column],
                errors="coerce",
            )
            mask |= values > relative_values

        if mask.any():
            replacement = rule.get("replacement")
            data.loc[mask, column] = np.nan if replacement is None else replacement

    return data


def remove_outliers(
    train: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Remove training outliers listed in the config."""
    config = CONFIG if config is None else config
    id_column = config["columns"]["id"]
    preprocessing = config["preprocessing"]

    if not preprocessing.get("remove_outliers", True):
        return train.copy()

    outlier_ids = preprocessing.get("outlier_ids", [])
    return train[~train[id_column].isin(outlier_ids)].copy()


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional domain features for the House Prices dataset."""
    data = data.copy()

    data["TotalSF"] = (
        data["TotalBsmtSF"].fillna(0)
        + data["1stFlrSF"].fillna(0)
        + data["2ndFlrSF"].fillna(0)
    )

    data["TotalBathrooms"] = (
        data["FullBath"].fillna(0)
        + 0.5 * data["HalfBath"].fillna(0)
        + data["BsmtFullBath"].fillna(0)
        + 0.5 * data["BsmtHalfBath"].fillna(0)
    )

    data["TotalPorchSF"] = (
        data["OpenPorchSF"].fillna(0)
        + data["EnclosedPorch"].fillna(0)
        + data["3SsnPorch"].fillna(0)
        + data["ScreenPorch"].fillna(0)
        + data["WoodDeckSF"].fillna(0)
    )

    data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
    data["GarageAge"] = data["YrSold"] - data["GarageYrBlt"]

    data["IsRemodeled"] = (data["YearRemodAdd"] != data["YearBuilt"]).astype(int)
    data["HasGarage"] = (data["GarageArea"].fillna(0) > 0).astype(int)
    data["HasBasement"] = (data["TotalBsmtSF"].fillna(0) > 0).astype(int)
    data["HasFireplace"] = (data["Fireplaces"].fillna(0) > 0).astype(int)
    data["HasPool"] = (data["PoolArea"].fillna(0) > 0).astype(int)
    data["HasPorch"] = (data["TotalPorchSF"].fillna(0) > 0).astype(int)

    data["QualitySF"] = data["OverallQual"] * data["TotalSF"]

    return data


def prepare_features_and_target(
    train: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and log-transformed target y."""
    config = CONFIG if config is None else config
    target_column = config["columns"]["target"]

    X = train.drop(columns=[target_column])
    y = np.log1p(train[target_column])
    return X, y


def get_feature_types(
    X: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """Split features into numeric and categorical feature names."""
    config = CONFIG if config is None else config
    id_column = config["columns"]["id"]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != id_column]

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features + [id_column]
    ]

    return numeric_features, categorical_features
