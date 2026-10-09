"""
Deterministic Synthetic Data Generator for QuantLab Pro.
Produces realistic OHLCV time-series data with fixed random seeds and provenance metadata.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash fingerprint of dataset values."""
    # Convert DataFrame to deterministic JSON representation for hashing
    records = df.to_dict(orient="records")
    serialized = json.dumps(records, default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_synthetic_ohlcv(
    symbol: str = "SYNTH_BTC",
    n_bars: int = 500,
    start_date: str = "2024-01-01",
    seed: int = 42,
    initial_price: float = 100.0,
    volatility: float = 0.015,
    trend: float = 0.0003,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate deterministic synthetic OHLCV time-series data.

    Args:
        symbol: Asset ticker name.
        n_bars: Number of trading periods (bars).
        start_date: Starting ISO date string (YYYY-MM-DD).
        seed: Random seed for 100% reproducibility.
        initial_price: Starting asset price.
        volatility: Daily price return volatility (std dev).
        trend: Daily drift term.

    Returns:
        Tuple of (DataFrame with OHLCV columns, Metadata dict).
    """
    rng = np.random.RandomState(seed)

    # Generate dates (daily business calendar or continuous)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(n_bars)]

    # Geometric Brownian Motion log returns
    log_returns = trend + volatility * rng.randn(n_bars)
    # Ensure log return first day is zero
    log_returns[0] = 0.0

    price_path = initial_price * np.exp(np.cumsum(log_returns))

    # Construct High, Low, Open, Close, Volume realistically
    open_prices = price_path * (1 + 0.002 * rng.randn(n_bars))
    close_prices = price_path

    # High must be >= max(Open, Close)
    high_extra = np.abs(0.008 * price_path * rng.randn(n_bars))
    high_prices = np.maximum(open_prices, close_prices) + high_extra

    # Low must be <= min(Open, Close)
    low_extra = np.abs(0.008 * price_path * rng.randn(n_bars))
    low_prices = np.minimum(open_prices, close_prices) - low_extra
    low_prices = np.maximum(0.01, low_prices)  # Ensure strictly positive

    # Volume: log-normal random variable scaled by volatility
    base_volume = 1_000_000
    volume = (base_volume * np.exp(0.5 * rng.randn(n_bars))).astype(int)
    volume = np.maximum(100, volume)

    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Open": np.round(open_prices, 4),
        "High": np.round(high_prices, 4),
        "Low": np.round(low_prices, 4),
        "Close": np.round(close_prices, 4),
        "Volume": volume,
    })

    fingerprint = compute_dataset_fingerprint(df)

    metadata = {
        "source": "synthetic",
        "symbol": symbol,
        "n_bars": n_bars,
        "seed": seed,
        "start_date": start_date,
        "end_date": df["Date"].iloc[-1],
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "initial_price": initial_price,
            "volatility": volatility,
            "trend": trend,
        }
    }

    return df, metadata
