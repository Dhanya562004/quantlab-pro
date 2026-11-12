"""
Feature Engineering Pipeline for Quantitative Alpha Modeling.
Reference Solution: Zero look-ahead leakage, strict backward windows, preserved API.
"""

import pandas as pd


def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Computes technical features from OHLCV market data.

    Columns produced:
    - returns
    - rolling_mean
    - rolling_std
    - z_score
    - volatility_ratio
    """
    features = pd.DataFrame(index=df.index)

    # 1. 1-bar returns (backward looking)
    features["returns"] = df["close"].pct_change()

    # 2. Backward-looking rolling statistics (center=False strictly enforced)
    rolling = df["close"].rolling(window=window, center=False)
    features["rolling_mean"] = rolling.mean()
    features["rolling_std"] = rolling.std()

    # 3. Normalized z-score relative to historical window
    features["z_score"] = (df["close"] - features["rolling_mean"]) / features["rolling_std"]

    # 4. Volatility ratio without centered look-ahead
    features["volatility_ratio"] = df["close"] / features["rolling_std"]

    # No bfill! Initial warmup periods remain NaN or are handled by caller
    return features


def prepare_dataset(
    df: pd.DataFrame,
    target_horizon: int = 1,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares features X and binary target y without look-ahead leakage.
    Target is 1 if close price at t + target_horizon > close price at t, else 0.
    """
    features = compute_features(df, window=window)

    # Target calculation aligned with horizon
    future_close = df["close"].shift(-target_horizon)
    target = (future_close > df["close"]).astype(int)
    target.name = "target"

    # Merge and drop NaNs (both initial window warmup and trailing horizon bars)
    aligned_df = features.copy()
    aligned_df["target"] = target
    # The last target_horizon rows must have NaN target because future is unknown
    aligned_df.iloc[-target_horizon:, aligned_df.columns.get_loc("target")] = float("nan")

    clean_df = aligned_df.dropna()

    X = clean_df.drop(columns=["target"])
    y = clean_df["target"].astype(int)

    return X, y

