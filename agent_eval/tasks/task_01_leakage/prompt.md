# Task Prompt: Fix Look-Ahead Data Leakage in Feature Pipeline

You are assigned to audit and fix a subtle look-ahead data leakage bug in `feature_pipeline.py`.

## Background & Problem
The quantitative research team observed that an ML trading strategy demonstrated suspiciously high backtest returns but failed immediately in out-of-sample forward testing. Code inspection suggests that features at time $t$ are leaking future price information from $t+1, t+2, \dots$.

## Exact Requirements

1. **File Location**:
   Modify `feature_pipeline.py` in place.

2. **Public API Signatures (Must Not Change)**:
   - `compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame`
   - `prepare_dataset(df: pd.DataFrame, target_horizon: int = 1, window: int = 20) -> tuple[pd.DataFrame, pd.Series]`

3. **Required Feature Columns**:
   The output of `compute_features` must contain exactly these five columns:
   - `'returns'`: 1-bar percentage change (`close.pct_change()`)
   - `'rolling_mean'`: Rolling mean of close prices over `window` bars.
   - `'rolling_std'`: Rolling standard deviation of close prices over `window` bars.
   - `'z_score'`: Normalized distance from rolling mean: `(close - rolling_mean) / rolling_std`
   - `'volatility_ratio'`: Ratio of close price to rolling standard deviation: `close / rolling_std`

4. **Strict Leakage-Free Rules**:
   - **No Centered Windows**: Rolling windows must be strictly backward-looking. Centered rolling windows (`center=True`) look forward in time and must not be used.
   - **No Backward Filling (`bfill`)**: Backward-filling missing values (`bfill` or `fillna(method='bfill')`) propagates future values backward into past rows. Do not use backward fill.
   - **Future Invariance**: If price observations at time $T+k$ (for any $k > 0$) are altered or perturbed, the computed features for all timestamps $t \le T$ must remain strictly unchanged.

5. **Target & Dataset Alignment**:
   - In `prepare_dataset`, compute features and align them with the forward-looking target.
   - Target is a binary indicator: `1` if close price at $t + \text{target\_horizon}$ is strictly greater than close price at $t$, else `0`.
   - The returned target must be a `pd.Series` named `'target'`.
   - The final `target_horizon` rows (which do not have a known future target) and initial warmup rows with `NaN` feature values must be dropped such that $X$ and $y$ have identical lengths and no `NaN` values.

