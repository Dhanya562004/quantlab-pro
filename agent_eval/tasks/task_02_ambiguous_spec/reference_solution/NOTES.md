# Technical Architecture & Convention Notes

## Technical Assumptions

### 1. Rolling Window Boundary Closure
- **Chosen Convention**: `closed='right'`.
- **Rationale**: In accordance with pandas defaults and standard quant research workflows, observation $T$ is included in the window $[T - \text{window} + 1, T]$ when evaluating signal states at close. If an execution system requires order entry at $T$'s open, signals can be explicitly lagged downstream using `.shift(1)`.

### 2. Warmup NaN Policy
- **Chosen Convention**: Preserve `NaN`s for initial $window$ bars (indices $0$ through $window - 1$). The first valid rolling volatility value is at index `window`. If `len(prices) <= window`, the returned series is entirely `NaN`.
- **Rationale**: Percentage returns (`prices.pct_change()`) inherently yields `NaN` at index 0. Consequently, computing rolling volatility over a window of $W$ returns requires $W$ valid return values, which requires $W + 1$ prices. The first bar with $W$ valid returns is at index $W$. When `len(prices) <= window`, at most $W - 1$ returns exist, so no full window can be computed and all values remain `NaN`. Warmup values should not be artificially synthesized, forward-filled, or backward-filled. Generating estimates on partial windows ($< window$) inflates estimation variance and introduces spurious trading triggers. Downstream model pipelines are expected to drop or mask warmup rows.

### 3. Annualization Basis Factor
- **Chosen Convention**: Scaling by $\sqrt{252}$ as default parameter (`trading_days=252`).
- **Rationale**: Standard equities and equity derivatives trade approximately 252 business days per calendar year. The function supports overriding `trading_days=365` for continuous crypto/forex instruments.

