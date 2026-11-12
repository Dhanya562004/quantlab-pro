
"""Hidden pytest grader for Task 01: Leakage."""

import inspect

import numpy as np
import pandas as pd
import pytest

from feature_pipeline import compute_features, prepare_dataset


EXPECTED_COLUMNS = [
    "returns",
    "rolling_mean",
    "rolling_std",
    "z_score",
    "volatility_ratio",
]


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, size=120)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close})


def assert_same_features(actual, expected, message):
    """Compare feature values and NaN positions."""
    assert actual.index.equals(expected.index), message
    assert actual.columns.equals(expected.columns), message

    np.testing.assert_array_equal(
        actual.isna().to_numpy(),
        expected.isna().to_numpy(),
        err_msg=message,
    )

    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
        err_msg=message,
    )


def test_public_api_signatures():
    """Verify exact parameter names, order, and defaults."""
    expected_features = [
        ("df", inspect.Parameter.empty),
        ("window", 20),
    ]
    expected_dataset = [
        ("df", inspect.Parameter.empty),
        ("target_horizon", 1),
        ("window", 20),
    ]

    for function, expected in [
        (compute_features, expected_features),
        (prepare_dataset, expected_dataset),
    ]:
        actual = inspect.signature(function).parameters
        assert list(actual) == [name for name, _ in expected]

        for name, default in expected:
            assert actual[name].default == default
            assert actual[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_feature_columns_and_types(sample_market_data):
    """Verify exact feature columns and DataFrame output."""
    features = compute_features(sample_market_data, window=20)

    assert isinstance(features, pd.DataFrame)
    assert list(features.columns) == EXPECTED_COLUMNS
    assert features.index.equals(sample_market_data.index)


def test_feature_formulas(sample_market_data):
    """Verify the exact mathematical formulas for all five features."""
    close = sample_market_data["close"]

    expected = pd.DataFrame(index=sample_market_data.index)
    expected["returns"] = close.pct_change()
    expected["rolling_mean"] = close.rolling(window=20).mean()
    expected["rolling_std"] = close.rolling(window=20).std()
    expected["z_score"] = (
        close - expected["rolling_mean"]
    ) / expected["rolling_std"]
    expected["volatility_ratio"] = (
        close / expected["rolling_std"]
    )

    actual = compute_features(sample_market_data, window=20)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1e-7,
        atol=1e-7,
    )


def test_future_perturbation_leakage(sample_market_data):
    """
    Changing observations from a future timestamp must not alter
    any feature values or NaN positions at earlier timestamps.
    """
    original = sample_market_data.copy()
    original_features = compute_features(original, window=20)

    perturbed = original.copy()
    perturbed.loc[60:, "close"] = (
        perturbed.loc[60:, "close"] * 2.5 + 50.0
    )
    perturbed_features = compute_features(perturbed, window=20)

    assert_same_features(
        original_features.iloc[:60],
        perturbed_features.iloc[:60],
        "Future price changes altered earlier features: look-ahead leakage detected.",
    )


def test_no_backward_fill_leakage(sample_market_data):
    """
    Warmup rows must remain missing, and changing a later observation
    must not alter earlier feature values.
    """
    original = sample_market_data.copy()
    original_features = compute_features(original, window=20)

    assert original_features.iloc[:19].isna().any().any(), (
        "Expected initial rolling warmup rows to contain NaN values."
    )
    assert original_features["rolling_mean"].iloc[:19].isna().all(), (
        "Initial rolling means must not be backward-filled."
    )

    perturbed = original.copy()
    perturbed.loc[19, "close"] *= 5.0
    perturbed_features = compute_features(perturbed, window=20)

    assert_same_features(
        original_features.iloc[:19],
        perturbed_features.iloc[:19],
        "Changing a future observation altered earlier features: backward-fill leakage detected.",
    )


@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_prepare_dataset_contract_and_alignment(sample_market_data, horizon):
    """
    Verify feature/target alignment, exact target values,
    warmup removal, and removal of unknown future targets.
    """
    X, y = prepare_dataset(
        sample_market_data,
        target_horizon=horizon,
        window=20,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == EXPECTED_COLUMNS
    assert y.name == "target"

    assert X.index.equals(y.index), "X and y indices must match exactly."
    assert len(X) == len(y)
    assert not X.isna().any().any(), "X contains NaN values."
    assert not y.isna().any(), "y contains NaN values."
    assert set(y.unique()).issubset({0, 1}), "Targets must be binary."
    assert pd.api.types.is_integer_dtype(y.dtype), (
        "Targets must have integer dtype."
    )

    features = compute_features(sample_market_data, window=20)
    future_close = sample_market_data["close"].shift(-horizon)

    valid_rows = features.notna().all(axis=1) & future_close.notna()
    expected_index = sample_market_data.index[valid_rows]

    assert X.index.equals(expected_index), (
        "Dataset contains invalid rows or drops valid rows."
    )

    expected_y = (
        future_close.loc[valid_rows]
        > sample_market_data.loc[valid_rows, "close"]
    ).astype(int)
    expected_y.name = "target"

    pd.testing.assert_series_equal(y, expected_y)
