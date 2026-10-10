"""
Unified Model Training Dispatcher & Artifact Generator for QuantLab Pro.
"""

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from quantlab.features.builder import DatasetSplits
from quantlab.models.baselines import (
    MajorityClassBaseline,
    SklearnLogisticRegressionModel,
)
from quantlab.models.pytorch_mlp import PyTorchMLPClassifier


class TrainedModelArtifact(BaseModel):
    """Artifact containing trained model, metadata, and training execution details."""
    model_type: str = Field(description="Name of model architecture used.")
    hyperparameters: dict[str, Any] = Field(description="Hyperparameters used for model training.")
    seed: int = Field(description="Random seed for reproducibility.")
    training_duration_sec: float = Field(description="Wall-clock training duration in seconds.")
    train_accuracy: float = Field(description="Accuracy achieved on training split.")
    val_accuracy: float = Field(description="Accuracy achieved on validation split.")
    training_history: list[dict[str, Any]] = Field(default_factory=list, description="Epoch loss/accuracy trace if deep learning.")
    model_object: Any = Field(default=None, exclude=True, description="Runtime model instance.")

    model_config = {"arbitrary_types_allowed": True}


def train_model(
    model_type: str,
    splits: DatasetSplits,
    hyperparameters: dict[str, Any] | None = None,
    seed: int = 42
) -> tuple[TrainedModelArtifact, Any]:
    """
    Unified trainer dispatching model training based on model_type.

    Supported model types:
    - 'majority_class'
    - 'logistic_regression'
    - 'pytorch_mlp'

    Args:
        model_type: Identifier string for model choice.
        splits: DatasetSplits containing X_train, y_train, X_val, y_val, X_test, y_test.
        hyperparameters: Dictionary of hyperparameter overrides.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (TrainedModelArtifact, Model instance)
    """
    if hyperparameters is None:
        hyperparameters = {}

    start_time = time.time()
    model_type_clean = model_type.lower().strip()

    if model_type_clean in ["majority_class", "baseline"]:
        model = MajorityClassBaseline()
        model.fit(splits.X_train, splits.y_train)
        training_history = []

    elif model_type_clean in ["logistic_regression", "logreg"]:
        c_val = float(hyperparameters.get("C", 1.0))
        max_iter = int(hyperparameters.get("max_iter", 1000))
        model = SklearnLogisticRegressionModel(C=c_val, max_iter=max_iter, seed=seed)
        model.fit(splits.X_train, splits.y_train)
        training_history = []

    elif model_type_clean in ["pytorch_mlp", "mlp", "nn"]:
        hidden_dim = int(hyperparameters.get("hidden_dim", 32))
        lr = float(hyperparameters.get("learning_rate", 0.005))
        epochs = int(hyperparameters.get("epochs", 40))
        batch_size = int(hyperparameters.get("batch_size", 32))
        dropout_rate = float(hyperparameters.get("dropout_rate", 0.2))

        model = PyTorchMLPClassifier(
            hidden_dim=hidden_dim,
            learning_rate=lr,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed
        )
        model.fit(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        training_history = model.training_history
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: ['majority_class', 'logistic_regression', 'pytorch_mlp']")

    duration = time.time() - start_time

    # Calculate train & val accuracies
    y_train_pred = model.predict(splits.X_train)
    train_acc = float(np.mean(y_train_pred == splits.y_train)) if len(splits.y_train) > 0 else 0.0

    y_val_pred = model.predict(splits.X_val)
    val_acc = float(np.mean(y_val_pred == splits.y_val)) if len(splits.y_val) > 0 else 0.0

    artifact = TrainedModelArtifact(
        model_type=model_type_clean,
        hyperparameters=hyperparameters,
        seed=seed,
        training_duration_sec=round(duration, 4),
        train_accuracy=round(train_acc, 4),
        val_accuracy=round(val_acc, 4),
        training_history=training_history,
        model_object=model,
    )

    return artifact, model


def save_model_artifact(
    experiment_id: str,
    artifact: TrainedModelArtifact,
    splits: DatasetSplits,
    output_dir: str = "artifacts/models"
) -> str:
    """
    Save model weights, hyperparameters, feature names, and scaler parameters linked to experiment_id.
    """
    import json
    from pathlib import Path

    import torch

    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "experiment_id": experiment_id,
        "model_type": artifact.model_type,
        "seed": artifact.seed,
        "hyperparameters": artifact.hyperparameters,
        "feature_names": splits.feature_names,
        "scaler_mean": splits.scaler_mean,
        "scaler_scale": splits.scaler_scale,
        "train_accuracy": artifact.train_accuracy,
        "val_accuracy": artifact.val_accuracy,
    }

    meta_file = dir_path / f"{experiment_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if artifact.model_type == "pytorch_mlp" and hasattr(artifact.model_object, "model") and artifact.model_object.model is not None:
        weights_file = dir_path / f"{experiment_id}.pt"
        torch.save(artifact.model_object.model.state_dict(), weights_file)
        meta["weights_path"] = str(weights_file)

    return str(meta_file)


def load_model_artifact(
    experiment_id: str,
    model_dir: str = "artifacts/models"
) -> dict[str, Any]:
    """
    Safely load experiment model artifact metadata and weights linked to experiment_id.
    """
    import json
    from pathlib import Path

    import torch

    from quantlab.models.pytorch_mlp import MLPNetwork, PyTorchMLPClassifier

    dir_path = Path(model_dir)
    meta_file = dir_path / f"{experiment_id}_meta.json"

    if not meta_file.exists():
        raise FileNotFoundError(f"No artifact found for experiment '{experiment_id}' at {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    weights_file = dir_path / f"{experiment_id}.pt"
    if meta.get("model_type") == "pytorch_mlp" and weights_file.exists():
        input_dim = len(meta.get("feature_names", []))
        hidden_dim = meta.get("hyperparameters", {}).get("hidden_dim", 32)
        net = MLPNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
        net.load_state_dict(torch.load(weights_file, weights_only=True))
        net.eval()
        model_wrapper = PyTorchMLPClassifier(hidden_dim=hidden_dim, seed=meta.get("seed", 42))
        model_wrapper.model = net
        model_wrapper.is_fitted = True
        meta["reconstructed_model"] = model_wrapper

    return meta

