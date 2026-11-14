"""
Known-bad solution 3: Uses the 'window - 1' assumption.
Flaw: Assumes that a rolling window of size W allows computing volatility starting at
index W - 1 (min_periods=window - 1) and that len(prices) == window yields a valid value,
ignoring that returns calculation loses the first observation (returns[0] is NaN).
"""

import numpy as np
import pandas as pd


def compute_rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    annualized: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")

    # FLAW: checks len < window, thinking len == window is sufficient to produce a value
    if len(prices) < window:
        return pd.Series(np.nan, index=prices.index, dtype=float)

    returns = prices.pct_change()

    # FLAW: sets min_periods=window - 1 to force an estimate at index window - 1
    rolling_std = returns.rolling(window=window, min_periods=window - 1, closed="right").std(ddof=1)

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
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")

    ema = prices.ewm(span=span, adjust=False).mean()
    momentum = (prices - ema) / ema
    return momentum

