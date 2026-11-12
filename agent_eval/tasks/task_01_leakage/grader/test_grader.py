"""Hidden pytest grader for Task 01: Leakage."""

import inspect

import numpy as np
import pandas as pd
import pytest
from feature_pipeline import compute_features, prepare_dataset


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
    returns = np.random.normal(0.0005, 0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close})


def test_public_api_signatures():
    """Verify function signatures and argument names match contract."""
    sig_feat = inspect.signature(compute_features)
    assert "df" in sig_feat.parameters
    assert "window" in sig_feat.parameters
    assert sig_feat.parameters["window"].default == 20

    sig_prep = inspect.signature(prepare_dataset)
    assert "df" in sig_prep.parameters
    assert "target_horizon" in sig_prep.parameters
    assert "window" in sig_prep.parameters
    assert sig_prep.parameters["target_horizon"].default == 1
    assert sig_prep.parameters["window"].default == 20


def test_feature_columns_and_types(sample_market_data):
    """Verify compute_features returns exact required columns."""
    features = compute_features(sample_market_data, window=20)
    assert isinstance(features, pd.DataFrame)
    expected_cols = ["returns", "rolling_mean", "rolling_std", "z_score", "volatility_ratio"]
    assert list(features.columns) == expected_cols


def test_future_perturbation_leakage(sample_market_data):
    """
    Core shifted-target / perturbation test:
    Altering future prices at t >= 60 MUST NOT alter feature values at t <= 50.
    Centered windows (center=True) or backward filling (bfill) fail this test.
    """
    df_orig = sample_market_data.copy()
    f_orig = compute_features(df_orig, window=20)

    # Perturb future data from row 60 onwards
    df_pert = df_orig.copy()
    df_pert.loc[60:, "close"] = df_pert.loc[60:, "close"] * 2.5 + 50.0
    f_pert = compute_features(df_pert, window=20)

    # Compare features prior to the perturbation cutoff (t=0 to t=50)
    # Check both warmup region and post-warmup region
    orig_slice = f_orig.iloc[:50]
    pert_slice = f_pert.iloc[:50]

    # Warmup NaNs should match or both be NaN; valid floats must be equal
    np.testing.assert_allclose(
        orig_slice.dropna().values,
        pert_slice.dropna().values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Lookahead leakage detected! Modifying future prices altered past feature values.",
    )


def test_no_backward_fill_leakage(sample_market_data):
    """
    Specific check against bfill leakage:
    1. Warmup rows prior to window must NOT be backward-filled with future values.
       Row 0 rolling_mean must be NaN.
    2. Changing future price at row 19 must not alter feature values at row 0.
    """
    df1 = sample_market_data.copy()
    f1 = compute_features(df1, window=20)

    # In a backward-only rolling window of size 20, row 0 cannot have a valid rolling mean
    assert pd.isna(f1.loc[0, "rolling_mean"]), (
        "Warmup leakage: row 0 rolling_mean is populated! Warmup values must not be backward-filled (bfill)."
    )

    df2 = sample_market_data.copy()
    df2.loc[19, "close"] = df2.loc[19, "close"] * 5.0
    f2 = compute_features(df2, window=20)

    # If bfill was used, row 0 changes based on row 19's perturbation
    if not pd.isna(f1.loc[0, "rolling_mean"]):
        assert f1.loc[0, "rolling_mean"] == f2.loc[0, "rolling_mean"], (
            "Backward-fill (bfill) leakage detected: altering row 19 changed row 0!"
        )


def test_prepare_dataset_contract_and_alignment(sample_market_data):
    """Verify prepare_dataset produces aligned X, y without NaNs or dimension mismatch."""
    X, y = prepare_dataset(sample_market_data, target_horizon=1, window=20)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert y.name == "target"
    assert len(X) == len(y)
    assert len(X) > 50
    assert not X.isna().any().any(), "X contains NaN values"
    assert not y.isna().any(), "y contains NaN values"
    assert set(y.unique()).issubset({0, 1}), "y must contain binary targets (0 or 1)"
