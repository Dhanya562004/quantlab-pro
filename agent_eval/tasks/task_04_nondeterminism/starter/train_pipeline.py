"""
PyTorch Neural Network Training Pipeline.
Currently suffers from multiple sources of nondeterminism.
"""

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_weight_hash(model: nn.Module) -> str:
    """Computes SHA-256 hash of model parameters."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def generate_synthetic_data(n_samples: int = 150):
    # Bug: Unseeded numpy generation
    X = np.random.randn(n_samples, 6).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] * 0.5) > 0.0).astype(np.float32).reshape(-1, 1)
    return X, y


def train_model(seed: int = 42, epochs: int = 15) -> dict[str, Any]:
    """
    Trains SimpleMLP model.
    Contains subtle sources of non-determinism across set order and DataLoader.
    """
    # Flaw: Missing RNG synchronization across libraries and backends

    # Flaw: Unordered set iteration leaks hash-seed non-determinism into feature columns
    raw_feature_set = {"volume", "volatility", "momentum", "spread", "skew", "kurtosis"}
    # Iteration over raw set without sorting:
    feature_names = list(raw_feature_set)

    X_np, y_np = generate_synthetic_data()

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    # Flaw: DataLoader shuffle without dedicated seeded generator
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleMLP(input_dim=len(feature_names))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    loss_history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / len(dataloader))

    weight_hash = compute_weight_hash(model)

    return {
        "model": model,
        "loss_history": loss_history,
        "weight_hash": weight_hash,
        "final_loss": loss_history[-1],
        "feature_names": feature_names,
    }
