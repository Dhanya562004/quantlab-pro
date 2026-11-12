"""Hidden pytest grader for Task 02: Ambiguous Spec & Boundary Conventions."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from indicators import compute_exponential_momentum, compute_rolling_volatility


@pytest.fixture
def sample_prices() -> pd.Series:
    np.random.seed(42)
    n = 60
    returns = np.random.normal(0.0002, 0.015, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), name="close")


def test_public_api_signatures():
    """Verify function signatures and argument names match contract."""
    sig_vol = inspect.signature(compute_rolling_volatility)
    assert list(sig_vol.parameters.keys()) == ["prices", "window", "annualized", "trading_days"]
    assert sig_vol.parameters["window"].default == 20
    assert sig_vol.parameters["annualized"].default is True
    assert sig_vol.parameters["trading_days"].default == 252

    sig_mom = inspect.signature(compute_exponential_momentum)
    assert list(sig_mom.parameters.keys()) == ["prices", "span"]
    assert sig_mom.parameters["span"].default == 14


def test_rolling_volatility_mathematical_precision(sample_prices):
    """Verify rolling volatility matches sample standard deviation (ddof=1) with sqrt scaling."""
    vol = compute_rolling_volatility(sample_prices, window=20, annualized=True, trading_days=252)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(sample_prices)

    # First window-1 values should be NaN
    assert vol.iloc[:19].isna().all(), "Warmup bars must be NaN"

    # Compute analytical reference value at index 25
    ret = sample_prices.pct_change()
    expected_val = ret.iloc[6:26].std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(vol.iloc[25], expected_val, rtol=1e-6, atol=1e-6)


def test_annualized_scaling(sample_prices):
    """Verify scaling factor sqrt(trading_days) applies accurately."""
    vol_annual = compute_rolling_volatility(sample_prices, window=10, annualized=True, trading_days=252)
    vol_raw = compute_rolling_volatility(sample_prices, window=10, annualized=False, trading_days=252)

    valid_mask = ~vol_raw.isna()
    ratio = vol_annual[valid_mask] / vol_raw[valid_mask]
    expected_ratio = np.sqrt(252)
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6, atol=1e-6)

    # Test custom trading_days (e.g. crypto 365)
    vol_crypto = compute_rolling_volatility(sample_prices, window=10, annualized=True, trading_days=365)
    ratio_crypto = vol_crypto[valid_mask] / vol_raw[valid_mask]
    np.testing.assert_allclose(ratio_crypto, np.sqrt(365), rtol=1e-6, atol=1e-6)


def test_zero_variance_and_flat_prices():
    """Verify constant flat prices return 0.0 without divide-by-zero or crash."""
    flat_prices = pd.Series([100.0] * 35)
    vol = compute_rolling_volatility(flat_prices, window=15, annualized=True)
    assert isinstance(vol, pd.Series)
    # Post-warmup values must evaluate to 0.0
    post_warmup = vol.iloc[15:]
    assert (post_warmup == 0.0).all(), "Flat price periods must have volatility of 0.0"


def test_invalid_inputs_and_short_series():
    """Verify error handling on negative prices and safe return on short series."""
    # Short series: len < window
    short_prices = pd.Series([10.0, 11.0, 12.0])
    vol_short = compute_rolling_volatility(short_prices, window=10)
    assert len(vol_short) == 3
    assert vol_short.isna().all()

    # Non-positive prices in rolling volatility
    with pytest.raises(ValueError):
        compute_rolling_volatility(pd.Series([10.0, 0.0, 12.0]))

    with pytest.raises(ValueError):
        compute_rolling_volatility(pd.Series([10.0, -5.0, 12.0]))

    # Non-positive prices in exponential momentum
    with pytest.raises(ValueError):
        compute_exponential_momentum(pd.Series([10.0, 0.0, 12.0]))


def test_exponential_momentum_calculation(sample_prices):
    """Verify exponential moving average distance calculation."""
    mom = compute_exponential_momentum(sample_prices, span=10)
    assert isinstance(mom, pd.Series)
    assert len(mom) == len(sample_prices)

    ema = sample_prices.ewm(span=10, adjust=False).mean()
    expected_mom = (sample_prices - ema) / ema
    np.testing.assert_allclose(mom.values, expected_mom.values, rtol=1e-6, atol=1e-6)


def test_notes_file_and_assumptions():
    """Verify NOTES.md exists and documents assumptions under ## Technical Assumptions."""
    notes_file = Path("NOTES.md")
    assert notes_file.exists(), "NOTES.md must exist in the workspace root."
    content = notes_file.read_text(encoding="utf-8")
    assert "## Technical Assumptions" in content, (
        "NOTES.md must contain a '## Technical Assumptions' heading."
    )
    # Check that boundary alignment, warmup NaN policy, and annualization are documented
    lower_content = content.lower()
    assert "closed" in lower_content or "right" in lower_content, "Must document window closure convention."
    assert "nan" in lower_content or "warmup" in lower_content, "Must document warmup NaN treatment."
    assert "annual" in lower_content or "252" in lower_content, "Must document annualization basis factor."
