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
