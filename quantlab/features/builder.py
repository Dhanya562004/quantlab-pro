"""
Leakage-Resistant Feature Engineering & Preprocessing for QuantLab Pro.
Ensures zero look-ahead bias and strict chronological train/validation/test splitting.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler


class FeatureConfig(BaseModel):
    """Configuration for feature engineering and prediction target."""
    lags: list[int] = Field(default=[1, 2, 3, 5], description="Lag periods for return features.")
    rolling_windows: list[int] = Field(default=[5, 10, 20], description="Rolling window sizes for volatility and SMA.")
    use_rsi: bool = Field(default=True, description="Compute Relative Strength Index feature.")
    use_macd: bool = Field(default=True, description="Compute MACD indicator feature.")
    target_horizon: int = Field(default=1, description="Number of bars forward to predict (h >= 1).")
    binary_target: bool = Field(default=True, description="Convert target into binary 1 (positive return) vs 0.")
    train_ratio: float = Field(default=0.6, description="Chronological train set fraction.")
    val_ratio: float = Field(default=0.2, description="Chronological validation set fraction.")
    test_ratio: float = Field(default=0.2, description="Chronological test set fraction.")


class DatasetSplits(BaseModel):
    """Encapsulates chronologically split and preprocessed datasets."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    df_aligned: Any | None = None

    model_config = {"arbitrary_types_allowed": True}


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index using Exponential Moving Average without look-ahead bias."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Compute MACD line and Signal line using EWM without look-ahead bias."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features_and_target(
    df: pd.DataFrame,
    config: FeatureConfig
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Construct leakage-resistant technical features and aligned prediction target.

    Strict Rule: All features at row index t are calculated using OHLCV data up to t ONLY.
    Target at row index t represents the return from t to t + config.target_horizon.

    Args:
        df: Input clean OHLCV DataFrame.
        config: Feature engineering configuration.

    Returns:
        Tuple of (X_df, y_series, feature_names, df_aligned)
    """
    df_feat = df.copy()
    close = df_feat["Close"]
    open_p = df_feat["Open"]
    high = df_feat["High"]
    low = df_feat["Low"]
    volume = df_feat["Volume"]

    feature_cols: list[str] = []

    # 1. Immediate returns & Lagged returns
    ret_1 = close.pct_change(1)
    df_feat["ret_lag_1"] = ret_1
    feature_cols.append("ret_lag_1")

    for k in config.lags:
        if k > 1:
            col_name = f"ret_lag_{k}"
            df_feat[col_name] = close.pct_change(k)
            feature_cols.append(col_name)

    # 2. Intra-bar volatility / spread features
    df_feat["hl_spread"] = (high - low) / (close + 1e-8)
    df_feat["co_spread"] = (close - open_p) / (open_p + 1e-8)
    feature_cols.extend(["hl_spread", "co_spread"])

    # 3. Rolling Volatility and Rolling SMA ratios
    for w in config.rolling_windows:
        vol_col = f"roll_vol_{w}"
        sma_col = f"sma_ratio_{w}"
        df_feat[vol_col] = ret_1.rolling(window=w).std()
        sma = close.rolling(window=w).mean()
        df_feat[sma_col] = (close - sma) / (sma + 1e-8)
        feature_cols.extend([vol_col, sma_col])

    # 4. Volume change feature
    df_feat["vol_change_1"] = volume.pct_change(1)
    feature_cols.append("vol_change_1")

    # 5. Technical Indicators (RSI, MACD)
    if config.use_rsi:
        df_feat["rsi_14"] = compute_rsi(close, window=14)
        feature_cols.append("rsi_14")

    if config.use_macd:
        macd_line, macd_signal = compute_macd(close)
        df_feat["macd_line"] = macd_line
        df_feat["macd_signal"] = macd_signal
        df_feat["macd_diff"] = macd_line - macd_signal
        feature_cols.extend(["macd_line", "macd_signal", "macd_diff"])

    # 6. Construct Target: Future Return from t to t + target_horizon
    future_return = (close.shift(-config.target_horizon) - close) / close
    if config.binary_target:
        # 1 if future return > 0 else 0
        target = (future_return > 0).astype(int)
    else:
        target = future_return

    df_feat["target"] = target
    df_feat["future_return"] = future_return

    # Drop NaNs resulting from rolling windows and target shifting
    valid_mask = df_feat[feature_cols + ["target"]].notna().all(axis=1)
    df_aligned = df_feat[valid_mask].copy().reset_index(drop=True)

    X = df_aligned[feature_cols]
    y = df_aligned["target"]

    return X, y, feature_cols, df_aligned


def create_chronological_splits(
    X: pd.DataFrame,
    y: pd.Series,
    df_aligned: pd.DataFrame,
    config: FeatureConfig
) -> DatasetSplits:
    """
    Split feature matrix X and target y into strict chronological Train, Validation, and Test sets.
    Fit StandardScaler ONLY on Training split, then transform Val and Test split.

    Args:
        X: Feature matrix DataFrame.
        y: Target Series.
        df_aligned: Aligned DataFrame containing 'Date'.
        config: FeatureConfig.

    Returns:
        DatasetSplits object with scaled numpy arrays and split metadata.
    """
    n_total = len(X)
    if n_total < 30:
        raise ValueError(f"Aligned dataset length ({n_total}) is too small for chronological splitting.")

    # Normalize split ratios to sum to 1.0
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    train_pct = config.train_ratio / total_ratio
    val_pct = config.val_ratio / total_ratio

    idx_train_end = int(np.floor(n_total * train_pct))
    idx_val_end = int(np.floor(n_total * (train_pct + val_pct)))

    # Ensure minimum sizes
    idx_train_end = max(10, min(idx_train_end, n_total - 10))
    idx_val_end = max(idx_train_end + 5, min(idx_val_end, n_total - 5))

    X_train_raw = X.iloc[:idx_train_end]
    y_train_raw = y.iloc[:idx_train_end].to_numpy()
    train_dates = df_aligned["Date"].iloc[:idx_train_end].tolist()

    X_val_raw = X.iloc[idx_train_end:idx_val_end]
    y_val_raw = y.iloc[idx_train_end:idx_val_end].to_numpy()
    val_dates = df_aligned["Date"].iloc[idx_train_end:idx_val_end].tolist()

    X_test_raw = X.iloc[idx_val_end:]
    y_test_raw = y.iloc[idx_val_end:].to_numpy()
    test_dates = df_aligned["Date"].iloc[idx_val_end:].tolist()

    # Standardize features: Fit ONLY on training split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return DatasetSplits(
        X_train=X_train_scaled,
        y_train=y_train_raw,
        X_val=X_val_scaled,
        y_val=y_val_raw,
        X_test=X_test_scaled,
        y_test=y_test_raw,
        feature_names=X.columns.tolist(),
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        df_aligned=df_aligned,
    )
