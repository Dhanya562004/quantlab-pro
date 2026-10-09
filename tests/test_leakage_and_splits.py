"""
Unit Tests for Leakage Prevention & Chronological Splits (QuantLab Pro).
Verifies strict temporal split boundaries, target alignment, and scaler fitting.
"""

import numpy as np

from quantlab.data.synthetic import generate_synthetic_ohlcv
from quantlab.features.builder import (
    FeatureConfig,
    build_features_and_target,
    create_chronological_splits,
)


def test_target_alignment_and_no_lookahead():
    df, _ = generate_synthetic_ohlcv(symbol="TEST", n_bars=300, seed=42)
    config = FeatureConfig(lags=[1, 2], target_horizon=1)
    X, y, feature_cols, df_aligned = build_features_and_target(df, config)

    # Verify that feature row t strictly uses data at t or earlier
    # Feature 'ret_lag_1' at row t must match (Close_t - Close_{t-1}) / Close_{t-1}
    idx = 10
    expected_ret_lag1 = (df_aligned["Close"].iloc[idx] - df_aligned["Close"].iloc[idx - 1]) / df_aligned["Close"].iloc[idx - 1]
    assert np.isclose(X["ret_lag_1"].iloc[idx], expected_ret_lag1, atol=1e-4)

    # Target at row t must represent future return: (Close_{t+1} - Close_t) / Close_t
    expected_future_ret = (df_aligned["Close"].iloc[idx + 1] - df_aligned["Close"].iloc[idx]) / df_aligned["Close"].iloc[idx]
    expected_binary_target = int(expected_future_ret > 0)
    assert y.iloc[idx] == expected_binary_target


def test_chronological_split_boundaries():
    df, _ = generate_synthetic_ohlcv(symbol="TEST", n_bars=300, seed=42)
    config = FeatureConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    X, y, feature_cols, df_aligned = build_features_and_target(df, config)
    splits = create_chronological_splits(X, y, df_aligned, config)

    # Verify dates do not overlap
    assert max(splits.train_dates) < min(splits.val_dates)
    assert max(splits.val_dates) < min(splits.test_dates)


def test_scaler_fit_on_train_only():
    df, _ = generate_synthetic_ohlcv(symbol="TEST", n_bars=300, seed=42)
    config = FeatureConfig()
    X, y, feature_cols, df_aligned = build_features_and_target(df, config)
    splits = create_chronological_splits(X, y, df_aligned, config)

    # Training split mean of scaled X_train must be ~0.0
    train_mean = np.mean(splits.X_train, axis=0)
    assert np.allclose(train_mean, 0.0, atol=1e-2)

    # Test split mean should NOT be forced to 0.0 (proving scaler wasn't fit on test set)
    test_mean = np.mean(splits.X_test, axis=0)
    assert not np.allclose(test_mean, 0.0, atol=1e-5)
