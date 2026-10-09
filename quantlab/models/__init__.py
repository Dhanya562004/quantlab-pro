"""Models module containing baselines, PyTorch MLP, and trainer dispatcher."""

from quantlab.models.baselines import (
    MajorityClassBaseline,
    SklearnLogisticRegressionModel,
)
from quantlab.models.pytorch_mlp import PyTorchMLPClassifier
from quantlab.models.trainer import TrainedModelArtifact, train_model

__all__ = [
    "MajorityClassBaseline",
    "PyTorchMLPClassifier",
    "SklearnLogisticRegressionModel",
    "TrainedModelArtifact",
    "train_model",
]
