"""
Unit Tests for Data Validation & Provenance (QuantLab Pro).
"""

import pandas as pd

from quantlab.data.synthetic import generate_synthetic_ohlcv
from quantlab.data.validation import validate_ohlcv_data


def test_synthetic_data_generation():
    df, meta = generate_synthetic_ohlcv(symbol="TEST_BTC", n_bars=150, seed=42)
    assert len(df) == 150
    assert meta["source"] == "synthetic"
    assert meta["symbol"] == "TEST_BTC"
    assert "fingerprint" in meta
    assert len(meta["fingerprint"]) == 64  # SHA-256 length


def test_deterministic_reproducibility():
    df1, meta1 = generate_synthetic_ohlcv(symbol="BTC", n_bars=100, seed=42)
    df2, meta2 = generate_synthetic_ohlcv(symbol="BTC", n_bars=100, seed=42)
    assert meta1["fingerprint"] == meta2["fingerprint"]
    pd.testing.assert_frame_equal(df1, df2)


def test_data_validation_pass():
    df, meta = generate_synthetic_ohlcv(symbol="BTC", n_bars=250, seed=100)
    report = validate_ohlcv_data(df)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.metrics["total_rows"] == 250


def test_data_validation_invalid_prices():
    df, meta = generate_synthetic_ohlcv(symbol="BTC", n_bars=150, seed=100)
    # Introduce invalid High < Low error
    df.loc[10, "High"] = df.loc[10, "Low"] - 5.0
    report = validate_ohlcv_data(df)
    assert report.is_valid is False
    assert any("Price boundary violations" in e for e in report.errors)


def test_data_validation_insufficient_sample():
    df, meta = generate_synthetic_ohlcv(symbol="BTC", n_bars=50, seed=100)
    report = validate_ohlcv_data(df, min_sample_size=100)
    assert report.is_valid is False
    assert any("Insufficient sample size" in e for e in report.errors)


def test_data_validation_out_of_order_dates():
    df, meta = generate_synthetic_ohlcv(symbol="BTC", n_bars=150, seed=100)
    # Swap dates to break monotonicity
    df.at[5, "Date"], df.at[6, "Date"] = df.at[6, "Date"], df.at[5, "Date"]
    report = validate_ohlcv_data(df)
    assert report.is_valid is False
    assert any("chronological" in e.lower() for e in report.errors)
