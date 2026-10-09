"""
Data Loader Utility for QuantLab Pro.
Handles CSV uploads and optional yfinance historical market data downloads with strict provenance tracking.
"""

import io
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from quantlab.data.synthetic import compute_dataset_fingerprint

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_csv_dataset(
    file_or_path: str | io.BytesIO | io.StringIO,
    symbol: str = "CSV_UPLOAD"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load and parse a CSV file containing market OHLCV data.

    Args:
        file_or_path: File path or file-like object.
        symbol: Custom label for the uploaded asset.

    Returns:
        Tuple of (DataFrame, Provenance Metadata Dict)
    """
    try:
        df = pd.read_csv(file_or_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {e!s}")

    # Standardize column names (case-insensitive strip)
    col_map = {c: c.strip().capitalize() for c in df.columns}
    # Map common variants
    for orig in df.columns:
        c_clean = orig.strip().lower()
        if c_clean in ["date", "timestamp", "time"]:
            col_map[orig] = "Date"
        elif c_clean in ["open"]:
            col_map[orig] = "Open"
        elif c_clean in ["high"]:
            col_map[orig] = "High"
        elif c_clean in ["low"]:
            col_map[orig] = "Low"
        elif c_clean in ["close", "adj close", "price"]:
            col_map[orig] = "Close"
        elif c_clean in ["volume", "vol"]:
            col_map[orig] = "Volume"

    df = df.rename(columns=col_map)

    # Verify required columns presence
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing mandatory required columns: {missing}. Expected {REQUIRED_COLUMNS}")

    df = df[REQUIRED_COLUMNS].copy()

    # Ensure Date column is formatted string
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert numerical columns
    for num_col in ["Open", "High", "Low", "Close", "Volume"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    fingerprint = compute_dataset_fingerprint(df)

    metadata = {
        "source": "uploaded_csv",
        "symbol": symbol,
        "n_bars": len(df),
        "start_date": df["Date"].iloc[0] if len(df) > 0 else "N/A",
        "end_date": df["Date"].iloc[-1] if len(df) > 0 else "N/A",
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return df, metadata


def load_yfinance_dataset(
    symbol: str = "AAPL",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Download historical market data from yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'MSFT', 'SPY').
        start_date: ISO start date.
        end_date: ISO end date.

    Returns:
        Tuple of (DataFrame, Provenance Metadata Dict)

    Raises:
        RuntimeError if yfinance is not available or download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance library is not installed. Please install it to fetch live market data.")

    try:
        ticker = yf.Ticker(symbol)
        df_raw = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if df_raw.empty:
            raise ValueError(f"No historical data returned for ticker symbol '{symbol}' between {start_date} and {end_date}.")

        df_raw = df_raw.reset_index()

        # Standardize column mapping
        df = pd.DataFrame()
        df["Date"] = pd.to_datetime(df_raw["Date"]).dt.strftime("%Y-%m-%d")
        df["Open"] = df_raw["Open"].round(4)
        df["High"] = df_raw["High"].round(4)
        df["Low"] = df_raw["Low"].round(4)
        df["Close"] = df_raw["Close"].round(4)
        df["Volume"] = df_raw["Volume"].astype(int)

        df = df.sort_values("Date").reset_index(drop=True)

        fingerprint = compute_dataset_fingerprint(df)

        metadata = {
            "source": "yfinance",
            "symbol": symbol.upper(),
            "n_bars": len(df),
            "start_date": df["Date"].iloc[0],
            "end_date": df["Date"].iloc[-1],
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return df, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to download market data for ticker '{symbol}': {e!s}")
