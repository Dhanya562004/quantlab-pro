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
    rolling_std = returns.rolling(window=window, min_periods=window, closed="right").std(ddof=1)

    # Handle flat / zero variance periods:
    # A price window of window + 1 prices produces window zero-return observations.
    # min_periods=window + 1 guarantees warmup bars (indices 0 to window - 1) remain NaN.
    rolling_min = prices.rolling(window=window + 1, min_periods=window + 1, closed="right").min()
    rolling_max = prices.rolling(window=window + 1, min_periods=window + 1, closed="right").max()
    flat_mask = (rolling_min == rolling_max) & (~rolling_min.isna())
    rolling_std = rolling_std.mask(flat_mask, 0.0)

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

