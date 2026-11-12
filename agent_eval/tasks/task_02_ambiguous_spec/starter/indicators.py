"""
Financial Technical Indicators Module.
Implement the functions below adhering to production standards and numerical robustness.
"""

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
    raise NotImplementedError("compute_rolling_volatility not implemented yet.")


def compute_exponential_momentum(
    prices: pd.Series,
    span: int = 14,
) -> pd.Series:
    """
    Computes normalized exponential moving average distance.
    """
    raise NotImplementedError("compute_exponential_momentum not implemented yet.")

