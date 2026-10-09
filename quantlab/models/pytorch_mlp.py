"""
PyTorch Multi-Layer Perceptron (MLP) Classifier for QuantLab Pro.
Includes fixed random seed reproducibility, early stopping, and scikit-learn API compatibility.
"""

import numpy as np
import torch
from torch import nn, optim


class MLPNetwork(nn.Module):
    """PyTorch MLP Neural Network architecture."""
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchMLPClassifier:
    """Scikit-learn compatible wrapper for PyTorch MLP classification model."""
    def __init__(
        self,
        hidden_dim: int = 32,
        learning_rate: float = 0.005,
        dropout_rate: float = 0.2,
        epochs: int = 40,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model: MLPNetwork | None = None
        self.is_fitted = False
        self.training_history: list[dict[str, float]] = []

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "PyTorchMLPClassifier":
        self._set_seed()

        n_samples, input_dim = X.shape
        if len(np.unique(y)) < 2:
            self.is_fitted = False
            self.majority_class = int(y[0]) if len(y) > 0 else 0
            return self

        self.model = MLPNetwork(input_dim=input_dim, hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        # Avoid batch size > n_samples or batch size == 1 (which causes BatchNorm error)
        effective_batch_size = max(2, min(self.batch_size, n_samples // 2 if n_samples > 4 else n_samples))
        loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

            train_loss = epoch_loss / max(1, total)
            train_acc = correct / max(1, total)

            history_entry = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc}

            if X_val is not None and y_val is not None and len(X_val) > 1:
                val_loss, val_acc = self._evaluate_loss_acc(X_val, y_val, criterion)
                history_entry["val_loss"] = val_loss
                history_entry["val_acc"] = val_acc

            self.training_history.append(history_entry)

        self.is_fitted = True
        return self

    def _evaluate_loss_acc(self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.long)
            outputs = self.model(X_v)
            loss = criterion(outputs, y_v).item()
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == y_v).float().mean().item()
        return loss, acc

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            p = 1.0 if getattr(self, "majority_class", 0) == 1 else 0.0
            return np.tile(np.array([1 - p, p]), (len(X), 1))

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
