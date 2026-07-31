from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from configs.config import get_artifact_paths, load_config
from src.data import load_data
from src.features import (
    add_features,
    apply_raw_data_fixes,
    get_feature_types,
    prepare_features_and_target,
    remove_outliers,
)
from src.models import create_preprocessor, regression_metrics_from_log_predictions
from src.torch_mlp import train_torch_mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyTorch MLP experiment on House Prices data."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config relative to project root or absolute path.",
    )
    return parser.parse_args()


def _to_dense_float32(matrix) -> np.ndarray:
    """Convert sklearn output to dense float32 numpy array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch_cfg = config["torch_mlp"]
    artifact_paths = get_artifact_paths(config)

    train, _, _ = load_data(config=config)
    train = apply_raw_data_fixes(train, config=config)
    train = remove_outliers(train, config=config)
    train = add_features(train)

    X, y_log = prepare_features_and_target(train, config=config)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y_log,
        test_size=torch_cfg["validation_size"],
        random_state=config["random_state"],
    )

    numeric_features, categorical_features = get_feature_types(X_train, config=config)
    preprocessor = create_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        config=config,
    )

    X_train_prepared = _to_dense_float32(preprocessor.fit_transform(X_train))
    X_valid_prepared = _to_dense_float32(preprocessor.transform(X_valid))

    requested_device = torch_cfg.get("device", "auto")
    device = None if requested_device == "auto" else requested_device

    model, history = train_torch_mlp(
        X_train=X_train_prepared,
        y_train=y_train.to_numpy(dtype=np.float32),
        X_valid=X_valid_prepared,
        y_valid=y_valid.to_numpy(dtype=np.float32),
        hidden_dims=tuple(torch_cfg["hidden_dims"]),
        dropout=torch_cfg["dropout"],
        learning_rate=torch_cfg["learning_rate"],
        weight_decay=torch_cfg["weight_decay"],
        batch_size=torch_cfg["batch_size"],
        epochs=torch_cfg["epochs"],
        patience=torch_cfg["patience"],
        seed=config["random_state"],
        device=device,
    )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_valid_tensor = torch.tensor(
            X_valid_prepared,
            dtype=torch.float32,
            device=device,
        )
        pred_log = model(X_valid_tensor).detach().cpu().numpy()

    metrics = regression_metrics_from_log_predictions(
        y_true_log=y_valid,
        y_pred_log=pred_log,
    )
    metrics = {f"valid_{key}": value for key, value in metrics.items()}
    metrics.update(
        {
            "epochs_requested": torch_cfg["epochs"],
            "epochs_ran": len(history.train_loss),
            "batch_size": torch_cfg["batch_size"],
            "learning_rate": torch_cfg["learning_rate"],
            "weight_decay": torch_cfg["weight_decay"],
            "dropout": torch_cfg["dropout"],
            "hidden_dims": "-".join(map(str, torch_cfg["hidden_dims"])),
            "patience": torch_cfg["patience"],
            "device": str(device),
            "best_valid_loss": min(history.valid_loss),
        }
    )

    artifact_paths["results_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(artifact_paths["torch_mlp_metrics"], index=False)

    print("PyTorch MLP validation metrics:")
    print(f"Validation RMSE log:   {metrics['valid_rmse_log']:.6f}")
    print(f"Validation MAE price:  {metrics['valid_mae_price']:.2f}")
    print(f"Validation RMSE price: {metrics['valid_rmse_price']:.2f}")
    print(f"Validation MAPE:       {metrics['valid_mape_percent']:.2f}%")
    print(f"Validation WAPE:       {metrics['valid_wape_percent']:.2f}%")
    print(f"Epochs ran:            {metrics['epochs_ran']} / {metrics['epochs_requested']}")
    print(f"Saved metrics to:      {artifact_paths['torch_mlp_metrics']}")


if __name__ == "__main__":
    main()
