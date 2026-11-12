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

    if len(prices) < window:
        return pd.Series(np.nan, index=prices.index, dtype=float)

    # 1. Percentage returns
    returns = prices.pct_change()

    # 2. Rolling standard deviation with ddof=1 (Bessel's correction)
    rolling_std = returns.rolling(window=window, min_periods=window, closed="right").std(ddof=1)

    # Handle flat / zero variance periods: replace NaN with 0.0 only where returns were constant
    # Check where rolling std is 0.0 or effectively 0.0
    rolling_std = rolling_std.fillna(np.nan)
    # For rows >= window, if prices were constant, std is 0.0
    rolling_min = prices.rolling(window=window, min_periods=window, closed="right").min()
    rolling_max = prices.rolling(window=window, min_periods=window, closed="right").max()
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

