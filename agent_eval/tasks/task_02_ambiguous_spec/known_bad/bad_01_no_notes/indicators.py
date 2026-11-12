"""Known-bad solution 1: Implements indicators but forgets NOTES.md."""

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
    if len(prices) < window:
        return pd.Series(np.nan, index=prices.index)
    ret = prices.pct_change()
    vol = ret.rolling(window=window).std(ddof=1)
    rolling_min = prices.rolling(window=window).min()
    rolling_max = prices.rolling(window=window).max()
    flat = (rolling_min == rolling_max) & (~rolling_min.isna())
    vol = vol.mask(flat, 0.0)
    if annualized:
        return vol * np.sqrt(trading_days)
    return vol


def compute_exponential_momentum(prices: pd.Series, span: int = 14) -> pd.Series:
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")
    ema = prices.ewm(span=span, adjust=False).mean()
    return (prices - ema) / ema

