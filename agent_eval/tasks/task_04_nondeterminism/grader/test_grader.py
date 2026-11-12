"""Hidden pytest grader for Task 04: Nondeterminism & Reproducibility."""

import torch.nn as nn
from train_pipeline import train_model


def test_api_contract_and_feature_sorting():
    """Verify return dictionary keys and alphabetical feature ordering."""
    res = train_model(seed=42, epochs=5)
    assert isinstance(res, dict)
    expected_keys = {"model", "loss_history", "weight_hash", "final_loss", "feature_names"}
    assert expected_keys.issubset(res.keys())

    assert isinstance(res["model"], nn.Module)
    assert isinstance(res["loss_history"], list)
    assert len(res["loss_history"]) == 5
    assert isinstance(res["weight_hash"], str)
    assert len(res["weight_hash"]) == 64  # SHA-256 length

    # Features must be sorted deterministically
    assert res["feature_names"] == sorted(res["feature_names"]), "Feature names must be sorted alphabetically"


def test_identical_runs_produce_identical_weights():
    """
    Core determinism test:
    Running training twice with identical seed=42 MUST produce bit-for-bit identical weight hashes.
    """
    res1 = train_model(seed=42, epochs=10)
    res2 = train_model(seed=42, epochs=10)

    assert res1["weight_hash"] == res2["weight_hash"], (
        f"Non-deterministic training! Run 1 hash={res1['weight_hash']} vs Run 2 hash={res2['weight_hash']}"
    )

    # Losses must be identical
    for e, (l1, l2) in enumerate(zip(res1["loss_history"], res2["loss_history"])):
        assert abs(l1 - l2) < 1e-6, f"Loss mismatch at epoch {e}: {l1} vs {l2}"


def test_different_seeds_produce_different_weights():
    """Verify that model is not hardcoding dummy static weights."""
    res_a = train_model(seed=42, epochs=8)
    res_b = train_model(seed=99, epochs=8)

    assert res_a["weight_hash"] != res_b["weight_hash"], (
        "Model outputs identical weights for different random seeds (possible hardcoding or dummy weights)."
    )


def test_model_actively_learns():
    """Verify model actually trains and reduces loss across epochs."""
    res = train_model(seed=42, epochs=15)
    initial_loss = res["loss_history"][0]
    final_loss = res["final_loss"]
    assert final_loss < initial_loss, f"Model failed to optimize: initial={initial_loss}, final={final_loss}"
