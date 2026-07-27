import numpy as np
import pandas as pd

from configs.config import ID_COLUMN, TARGET_COLUMN


OUTLIER_IDS = [524, 1299]


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Remove known outliers from the training dataset."""
    train_clean = train[~train[ID_COLUMN].isin(OUTLIER_IDS)].copy()
    return train_clean


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for the House Prices dataset."""
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


def prepare_features_and_target(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and log-transformed target y."""
    X = train.drop(columns=[TARGET_COLUMN])
    y = np.log1p(train[TARGET_COLUMN])

    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split features into numeric and categorical lists."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != ID_COLUMN]

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features + [ID_COLUMN]
    ]

    return numeric_features, categorical_features