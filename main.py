import joblib
import pandas as pd

from configs.config import (
    ID_COLUMN,
    RESULTS_DIR,
    SUBMISSIONS_DIR,
)

from src.data import load_data
from src.features import (
    add_features,
    get_feature_types,
    prepare_features_and_target,
    remove_outliers,
)
from src.models import create_final_model, evaluate_model
from src.submission import create_submission, validate_submission


def main() -> None:
    """Run full House Prices training and submission pipeline."""
    train, test, sample_submission = load_data()

    train_clean = remove_outliers(train)

    train_fe = add_features(train_clean)
    test_fe = add_features(test)

    X, y = prepare_features_and_target(train_fe)
    numeric_features, categorical_features = get_feature_types(X)

    final_model = create_final_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    metrics = evaluate_model(final_model, X, y)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = RESULTS_DIR / "final_model_metrics.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    print("Final model CV metrics:")
    print(f"CV RMSE log mean: {metrics['cv_rmse_log_mean']:.6f}")
    print(f"CV RMSE log std: {metrics['cv_rmse_log_std']:.6f}")
    print(f"Saved metrics to: {metrics_path}")

    final_model.fit(X, y)

    model_path = RESULTS_DIR / "final_lasso_catboost_ensemble.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved model to: {model_path}")

    final_submission = create_submission(
        model=final_model,
        test_features=test_fe,
        test_ids=test[ID_COLUMN],
    )

    validate_submission(
        submission=final_submission,
        sample_submission=sample_submission,
    )

    submission_path = SUBMISSIONS_DIR / "final_lasso_catboost_ensemble_submission.csv"
    final_submission.to_csv(submission_path, index=False)

    print(f"Saved submission to: {submission_path}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()