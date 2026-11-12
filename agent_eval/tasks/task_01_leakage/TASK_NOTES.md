# Task 01: Subtle Data Leakage in Feature Engineering Pipeline

## Overview
This task assesses whether an agent can detect and eradicate subtle look-ahead data leakage in a quantitative feature engineering module without violating public function signatures or breaking downstream API contracts.

## Sources of Difficulty (Why This Task Is Hard)

### 1. Dual-Channel Look-Ahead Leakage (Subtle, Not Obvious)
Most coding benchmarks only test for trivial leakage (e.g., target shifted in the wrong direction or calling `scaler.fit(full_dataset)`). In this task, leakage occurs across two distinct mechanisms:
- **Centered Window Look-Ahead:** The rolling mean and volatility ratio features use `rolling(window=window, center=True)`. In pandas, `center=True` centers the window over the current row, silently pulling data from $t+1 \dots t + \lfloor window/2 \rfloor$ into time $t$. Agents often overlook `center=True` when skimming rolling calculations.
- **Backward Fill (`bfill`) Propagation:** To handle initial warmup `NaN` values produced by rolling windows, the starter code executes `features.bfill()`. Backward filling takes the first valid future value and pastes it backward into earlier timestamps. As a result, the feature values at $t=0$ depend directly on future market prices at $t=window-1$. A future perturbation at $t=15$ silently alters $t=0$, causing catastrophic leakage that ruins live trading models.

### 2. Strict API Contract & Invariant Preservation
In production quantitative infrastructure, upstream bug fixes cannot break downstream consumers. The agent must:
- Preserve the exact signatures:
  `compute_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame`
  `prepare_dataset(df: pd.DataFrame, target_horizon: int = 1, window: int = 20) -> tuple[pd.DataFrame, pd.Series]`
- Preserve exact column names: `['returns', 'rolling_mean', 'rolling_std', 'z_score', 'volatility_ratio']`.
- Maintain clean index alignment between feature matrix $X$ and binary direction target $y$ without returning `NaN`s or mismatched dimensions.
Naive agents frequently attempt to "fix" leakage by dropping columns, altering function signatures, or introducing arbitrary row-dropping logic that breaks the API contract.

