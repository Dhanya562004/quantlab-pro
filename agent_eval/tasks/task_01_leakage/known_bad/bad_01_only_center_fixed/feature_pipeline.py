"""Known-bad solution 1: Fixes center=False, but keeps bfill()."""

import pandas as pd


def compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["returns"] = df["close"].pct_change()

    # Fixed center=False
    rolling = df["close"].rolling(window=window, center=False)
    features["rolling_mean"] = rolling.mean()
    features["rolling_std"] = rolling.std()
    features["z_score"] = (df["close"] - features["rolling_mean"]) / features["rolling_std"]
    features["volatility_ratio"] = df["close"] / features["rolling_std"]

    # FLAW: Kept bfill(), which leaks future information backward into past rows!
    features = features.bfill()
    return features


def prepare_dataset(
    df: pd.DataFrame,
    target_horizon: int = 1,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    features = compute_features(df, window=window)
    future_close = df["close"].shift(-target_horizon)
    target = (future_close > df["close"]).astype(int)
    target.name = "target"
    aligned_df = features.copy()
    aligned_df["target"] = target
    clean = aligned_df.dropna()
    return clean.drop(columns=["target"]), clean["target"].astype(int)

