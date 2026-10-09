"""
Quantitative Data Validation Module for QuantLab Pro.
Performs comprehensive schema, pricing, temporal ordering, and distributional checks.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class ValidationReport(BaseModel):
    """Structured report of quantitative data validation checks."""
    is_valid: bool = Field(description="True if no blocking errors found.")
    errors: list[str] = Field(default_factory=list, description="Critical blocking validation errors.")
    warnings: list[str] = Field(default_factory=list, description="Non-blocking data quality warnings.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Summary statistics and data metrics.")
    audit_trail: list[dict[str, Any]] = Field(default_factory=list, description="Step-by-step audit verification entries.")


def validate_ohlcv_data(
    df: pd.DataFrame,
    min_sample_size: int = 100,
    outlier_threshold_std: float = 5.0
) -> ValidationReport:
    """
    Run comprehensive quantitative data validation on an OHLCV dataset.
    
    Checks:
    1. Schema & required columns
    2. Missing / NaN values
    3. Chronological sorting & date monotonicity
    4. Duplicate timestamps
    5. Non-positive prices (Close, High, Low, Open <= 0)
    6. Price boundary integrity (High < Low, High < Open, Low > Close, etc.)
    7. Invalid volume (Volume < 0)
    8. Outliers (excessive single-period price percentage jumps)
    9. Minimum sample size adequacy
    
    Args:
        df: Input DataFrame.
        min_sample_size: Minimum required rows for modeling.
        outlier_threshold_std: Standard deviation multiplier for return outlier detection.
        
    Returns:
        ValidationReport object containing pass status, errors, warnings, and audit log.
    """
    errors: list[str] = []
    warnings: list[str] = []
    audit_trail: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    
    # 1. Required Columns Check
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        errors.append(f"Missing mandatory column(s): {missing_cols}")
        audit_trail.append({"step": "Schema Validation", "status": "FAIL", "message": f"Missing columns: {missing_cols}"})
        return ValidationReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            audit_trail=audit_trail,
        )
    audit_trail.append({"step": "Schema Validation", "status": "PASS", "message": "All mandatory columns present."})
    
    n_rows = len(df)
    metrics["total_rows"] = n_rows
    
    # 2. Sample Size Check
    if n_rows < min_sample_size:
        errors.append(f"Insufficient sample size: dataset has {n_rows} rows, minimum required is {min_sample_size}.")
        audit_trail.append({"step": "Sample Size Check", "status": "FAIL", "message": f"Rows={n_rows} < Minimum={min_sample_size}"})
    elif n_rows < 200:
        warnings.append(f"Small dataset size ({n_rows} rows). Model training may be prone to overfitting.")
        audit_trail.append({"step": "Sample Size Check", "status": "WARNING", "message": f"Rows={n_rows} < Recommended=200"})
    else:
        audit_trail.append({"step": "Sample Size Check", "status": "PASS", "message": f"Sufficient sample size ({n_rows} rows)."})

    # 3. Missing / NaN Values
    nan_counts = df[required_cols].isna().sum().to_dict()
    total_nans = sum(nan_counts.values())
    metrics["nan_counts"] = nan_counts
    
    if total_nans > 0:
        if nan_counts["Close"] > 0 or nan_counts["Date"] > 0:
            errors.append(f"Missing values detected in critical columns: {nan_counts}")
            audit_trail.append({"step": "Missing Values Check", "status": "FAIL", "message": f"NaNs found: {nan_counts}"})
        else:
            warnings.append(f"Missing values detected in non-critical columns: {nan_counts}. Imputation required.")
            audit_trail.append({"step": "Missing Values Check", "status": "WARNING", "message": f"NaNs found: {nan_counts}"})
    else:
        audit_trail.append({"step": "Missing Values Check", "status": "PASS", "message": "Zero missing values."})

    # 4. Temporal Ordering & Monotonicity
    date_series = pd.to_datetime(df["Date"], errors="coerce")
    if date_series.isna().any():
        errors.append("Unparseable timestamps present in 'Date' column.")
        audit_trail.append({"step": "Date Parsing", "status": "FAIL", "message": "Invalid date strings found."})
    else:
        metrics["start_date"] = date_series.min().strftime("%Y-%m-%d")
        metrics["end_date"] = date_series.max().strftime("%Y-%m-%d")
        
        # Check monotonicity: strictly increasing date ordering
        diffs = date_series.diff().dropna()
        is_strictly_increasing = (diffs.dt.total_seconds() > 0).all() if len(diffs) > 0 else True
        
        if not is_strictly_increasing:
            errors.append("Timestamps are not strictly in chronological ascending order.")
            audit_trail.append({"step": "Chronological Ordering", "status": "FAIL", "message": "Date column is out of order."})
        else:
            audit_trail.append({"step": "Chronological Ordering", "status": "PASS", "message": "Strictly ascending chronological order."})
            
        # Check duplicate timestamps
        duplicates = date_series.duplicated().sum()
        metrics["duplicate_timestamps"] = int(duplicates)
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate timestamp entries.")
            audit_trail.append({"step": "Duplicate Timestamps Check", "status": "FAIL", "message": f"{duplicates} duplicates."})
        else:
            audit_trail.append({"step": "Duplicate Timestamps Check", "status": "PASS", "message": "No duplicate timestamps."})

    # 5. Non-positive Prices & Price Boundary Integrity
    price_cols = ["Open", "High", "Low", "Close"]
    non_positive = (df[price_cols] <= 0).sum().to_dict()
    total_non_pos = sum(non_positive.values())
    metrics["non_positive_prices"] = total_non_pos
    
    if total_non_pos > 0:
        errors.append(f"Invalid non-positive price values detected: {non_positive}")
        audit_trail.append({"step": "Price Validity Check", "status": "FAIL", "message": f"Non-positive prices: {non_positive}"})
    else:
        audit_trail.append({"step": "Price Validity Check", "status": "PASS", "message": "All prices strictly positive."})

    # High < Low, High < Open, High < Close, Low > Open, Low > Close
    invalid_high_low = (df["High"] < df["Low"]).sum()
    invalid_high_open = (df["High"] < df["Open"]).sum()
    invalid_high_close = (df["High"] < df["Close"]).sum()
    invalid_low_open = (df["Low"] > df["Open"]).sum()
    invalid_low_close = (df["Low"] > df["Close"]).sum()
    
    total_boundary_violations = invalid_high_low + invalid_high_open + invalid_high_close + invalid_low_open + invalid_low_close
    metrics["price_boundary_violations"] = int(total_boundary_violations)
    
    if total_boundary_violations > 0:
        errors.append(f"Price boundary violations found: High < Low ({invalid_high_low}), High < Open ({invalid_high_open}), Low > Close ({invalid_low_close}).")
        audit_trail.append({"step": "Price Boundary Integrity", "status": "FAIL", "message": f"{total_boundary_violations} boundary violations."})
    else:
        audit_trail.append({"step": "Price Boundary Integrity", "status": "PASS", "message": "High >= max(Open, Close) and Low <= min(Open, Close) holds true."})

    # 6. Volume Integrity Check
    negative_vol = (df["Volume"] < 0).sum()
    metrics["negative_volume"] = int(negative_vol)
    if negative_vol > 0:
        errors.append(f"Invalid negative volume count: {negative_vol} rows.")
        audit_trail.append({"step": "Volume Validity Check", "status": "FAIL", "message": f"{negative_vol} negative volume rows."})
    else:
        audit_trail.append({"step": "Volume Validity Check", "status": "PASS", "message": "Volume is non-negative."})

    # 7. Outliers Check (Excessive Return Jump)
    if n_rows >= 5:
        pct_returns = df["Close"].pct_change().dropna()
        mean_ret = pct_returns.mean()
        std_ret = pct_returns.std()
        
        if std_ret > 0:
            z_scores = (pct_returns - mean_ret) / std_ret
            outlier_count = (z_scores.abs() > outlier_threshold_std).sum()
            metrics["outlier_count"] = int(outlier_count)
            if outlier_count > 0:
                warnings.append(f"Detected {outlier_count} price return jump outliers (> {outlier_threshold_std} std dev).")
                audit_trail.append({"step": "Outlier Detection", "status": "WARNING", "message": f"{outlier_count} price outliers."})
            else:
                audit_trail.append({"step": "Outlier Detection", "status": "PASS", "message": "No extreme return outliers detected."})

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
        audit_trail=audit_trail,
    )
