# Task Prompt: Implement Financial Technical Indicators with Ambiguous Boundary Conventions

You are asked to implement technical indicator functions in `indicators.py` and document your technical assumptions in `NOTES.md`.

## Functions to Implement in `indicators.py`

### 1. `compute_rolling_volatility(prices: pd.Series, window: int = 20, annualized: bool = True, trading_days: int = 252) -> pd.Series`
- Computes rolling realized volatility of percentage returns (`prices.pct_change()`).
- Uses sample standard deviation with Bessel's correction (`ddof=1`).
- If `annualized=True`, multiplies rolling standard deviation by $\sqrt{\text{trading\_days}}$. If `False`, returns unannualized rolling standard deviation.
- When `len(prices) < window`, returns a `pd.Series` of `NaN`s with the identical index and length as `prices`.
- Handles zero variance / flat price periods gracefully: if prices are constant over the window, rolling volatility must evaluate to `0.0` (not `NaN` or unhandled division).
- If `prices` contains any non-positive values ($\le 0$), raises `ValueError("Prices must be strictly positive")`.

### 2. `compute_exponential_momentum(prices: pd.Series, span: int = 14) -> pd.Series`
- Computes normalized exponential moving average distance:
  $$\text{momentum} = \frac{\text{prices} - \text{EMA}}{\text{EMA}}$$
  where $\text{EMA} = \text{prices.ewm}(span=span, adjust=False).\text{mean}()$.
- If `prices` contains any non-positive values ($\le 0$), raises `ValueError("Prices must be strictly positive")`.
- Returns a `pd.Series` with the same index as `prices`.

## Specification Ambiguity & Required Documentation

In financial time-series engineering, rolling window definitions carry inherent ambiguities across trading platforms:
1. **Window Alignment / Boundary Closure**: Does timestamp $T$ include observation $T$ (`closed='right'`) or only historical completed bars up to $T-1$ (`closed='left'`)? In this repository, the required standard is `closed='right'` (standard pandas rolling default).
2. **Warmup Policy**: How should the initial $window - 1$ bars be treated? Standard production practice is preserving `NaN` values rather than fabricating or extrapolating artificial warmup values.
3. **Annualization Basis**: Standard equity trading year assumes `trading_days=252`.

You **MUST** create a file named `NOTES.md` in the workspace root containing a section heading `## Technical Assumptions` that explicitly states and explains:
- The chosen rolling window closure convention (`closed='right'`).
- The warmup `NaN` policy (preserving `NaN`s for initial bars $< window$).
- The annualization basis factor ($\sqrt{252}$ trading days).

