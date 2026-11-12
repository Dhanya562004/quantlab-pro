"""Known-bad solution 2: Returns dummy hardcoded model and static weights."""

from typing import Any

import torch.nn as nn


def train_model(seed: int = 42, epochs: int = 15) -> dict[str, Any]:
    # FLAW: Hardcoded dummy model and static loss
    model = nn.Sequential(nn.Linear(6, 1))
    return {
        "model": model,
        "loss_history": [0.5] * epochs,
        "weight_hash": "a" * 64,
        "final_loss": 0.5,
        "feature_names": ["kurtosis", "momentum", "skew", "spread", "volatility", "volume"],
    }
