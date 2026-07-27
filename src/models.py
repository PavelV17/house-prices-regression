import pandas as pd

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingRegressor

from configs.config import N_SPLITS, RANDOM_STATE, SCORING


def create_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def create_final_model(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create final Lasso + CatBoost ensemble model."""
    preprocessor = create_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    lasso_model = Lasso(
        alpha=0.0005,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )

    catboost_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=5,
        loss_function="RMSE",
        random_state=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )

    ensemble_model = VotingRegressor(
        estimators=[
            ("lasso", lasso_model),
            ("catboost", catboost_model),
        ],
        weights=[0.6, 0.4],
    )

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ensemble_model),
        ]
    )

    return final_pipeline


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Evaluate model using cross-validation."""
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = -cross_val_score(
        model,
        X,
        y,
        scoring=SCORING,
        cv=cv,
        n_jobs=1,
    )

    return {
        "cv_rmse_log_mean": scores.mean(),
        "cv_rmse_log_std": scores.std(),
    }