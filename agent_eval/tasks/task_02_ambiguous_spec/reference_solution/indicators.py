"""
Financial Technical Indicators Module.
Reference Solution: Numerically stable, handles edge cases, strict Bessel correction.
"""

import numpy as np
import pandas as pd


def compute_rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    annualized: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Computes rolling realized volatility of returns.
    """
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")

    # When len(prices) <= window, pct_change has at most window - 1 valid returns,
    # so a rolling window of size window cannot be formed.
    if len(prices) <= window:
        return pd.Series(np.nan, index=prices.index, dtype=float)

    # 1. Percentage returns (returns.iloc[0] is NaN)
    returns = prices.pct_change()

    # 2. Rolling standard deviation with ddof=1 (Bessel's correction).
    # Since returns.iloc[0] is NaN, min_periods=window ensures the first valid
    # rolling volatility is at index window (requires window + 1 prices).
    # Note: Constant prices produce returns of 0.0, which evaluate natively to std=0.0.
    rolling_std = returns.rolling(window=window, min_periods=window, closed="right").std(ddof=1)

    if annualized:
        return rolling_std * np.sqrt(trading_days)
    return rolling_std


def compute_exponential_momentum(
    prices: pd.Series,
    span: int = 14,
) -> pd.Series:
    """
    Computes normalized exponential moving average distance.
    """
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")

    ema = prices.ewm(span=span, adjust=False).mean()
    momentum = (prices - ema) / ema
    return momentum

