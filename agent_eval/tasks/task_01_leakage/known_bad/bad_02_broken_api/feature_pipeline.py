"""Known-bad solution 2: Breaks public API signature and column names."""

import pandas as pd


def compute_features(df: pd.DataFrame, window: int = 20, drop_na: bool = True) -> pd.DataFrame:
    # FLAW: Renames required columns and adds new non-default signature parameter
    features = pd.DataFrame(index=df.index)
    features["ret_1d"] = df["close"].pct_change()
    rolling = df["close"].rolling(window=window, center=False)
    features["mean"] = rolling.mean()
    features["std"] = rolling.std()
    return features


def prepare_dataset(df: pd.DataFrame, target_horizon: int = 1, window: int = 20) -> dict:
    # FLAW: Returns dict instead of (X, y) tuple
    features = compute_features(df, window=window)
    return {"features": features}

