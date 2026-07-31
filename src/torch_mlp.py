from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.config import RANDOM_STATE


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Make PyTorch training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPRegressorNet(nn.Module):
    """Small fully-connected network for tabular regression experiments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class TorchTrainingHistory:
    train_loss: list[float]
    valid_loss: list[float]


def train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 15,
    seed: int = RANDOM_STATE,
    device: str | None = None,
) -> tuple[MLPRegressorNet, TorchTrainingHistory]:
    """Train a PyTorch MLP on already-preprocessed tabular features."""
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = MLPRegressorNet(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = TorchTrainingHistory(train_loss=[], valid_loss=[])
    best_valid_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_predictions = model(X_valid_tensor)
            valid_loss = criterion(valid_predictions, y_valid_tensor).item()

        history.train_loss.append(float(np.mean(train_losses)))
        history.valid_loss.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
