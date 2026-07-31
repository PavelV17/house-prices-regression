from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from configs.config import get_artifact_paths, load_config
from src.inference import create_submission_from_model, run_inference
from src.train import train_final_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train House Prices model and create Kaggle submission."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config relative to project root or absolute path.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name saved to results/experiments.csv.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip training, load saved model and create submission only.",
    )
    return parser.parse_args()


def print_metrics(metrics: dict[str, float]) -> None:
    """Print final model metrics in a readable format."""
    print("Final model CV metrics:")
    print(f"CV RMSE log mean: {metrics['cv_rmse_log_mean']:.6f}")
    print(f"CV RMSE log std:  {metrics['cv_rmse_log_std']:.6f}")
    print(f"OOF RMSE log:     {metrics['oof_rmse_log']:.6f}")
    print(f"OOF MAE price:    {metrics['oof_mae_price']:.2f}")
    print(f"OOF RMSE price:   {metrics['oof_rmse_price']:.2f}")
    print(f"OOF MAPE:         {metrics['oof_mape_percent']:.2f}%")
    print(f"OOF WAPE:         {metrics['oof_wape_percent']:.2f}%")


def main() -> None:
    """Run full House Prices training and submission pipeline."""
    args = parse_args()
    config = load_config(args.config)
    experiment_name = args.experiment or config.get("experiment", {}).get("name")
    artifact_paths = get_artifact_paths(config)

    if args.inference_only:
        submission = run_inference(config=config)
        print(f"Saved submission to: {artifact_paths['final_submission']}")
        print(f"Submission rows: {len(submission)}")
        print("Inference completed successfully.")
        return

    training_result = train_final_model(
        save_artifacts=True,
        config=config,
        experiment_name=experiment_name,
    )

    artifact_paths["submissions_dir"].mkdir(parents=True, exist_ok=True)
    final_submission = create_submission_from_model(
        model=training_result.model,
        test_features=training_result.test_features,
        test_ids=training_result.test_ids,
        sample_submission=training_result.sample_submission,
        config=config,
    )
    final_submission.to_csv(artifact_paths["final_submission"], index=False)

    print(f"Experiment: {experiment_name}")
    print_metrics(training_result.metrics)
    print(f"Saved metrics to: {artifact_paths['final_metrics']}")
    print(f"Saved model to: {artifact_paths['final_model']}")
    print(f"Saved submission to: {artifact_paths['final_submission']}")
    print(f"Saved experiment log to: {artifact_paths['experiments']}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
