"""Known-bad solution 2: Has NOTES.md but crashes on short series and zero-variance."""

import numpy as np
import pandas as pd


def compute_rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    annualized: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    # FLAW: Crashes on short series with index error or unhandled slice, and doesn't validate <= 0
    if len(prices) < window:
        raise IndexError("Price series too short!")

    ret = prices.pct_change()
    # Uses ddof=0 and fails flat prices handling
    vol = ret.rolling(window=window).std(ddof=0)
    if annualized:
        return vol * np.sqrt(trading_days)
    return vol


def compute_exponential_momentum(prices: pd.Series, span: int = 14) -> pd.Series:
    # FLAW: Doesn't validate positive prices
    ema = prices.ewm(span=span, adjust=False).mean()
    return (prices - ema) / ema

