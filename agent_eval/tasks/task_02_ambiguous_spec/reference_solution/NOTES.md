# Technical Architecture & Convention Notes

## Technical Assumptions

### 1. Rolling Window Boundary Closure
- **Chosen Convention**: `closed='right'`.
- **Rationale**: In accordance with pandas defaults and standard quant research workflows, observation $T$ is included in the window $[T - \text{window} + 1, T]$ when evaluating signal states at close. If an execution system requires order entry at $T$'s open, signals can be explicitly lagged downstream using `.shift(1)`.

### 2. Warmup NaN Policy
- **Chosen Convention**: Preserve `NaN`s for initial $window - 1$ periods.
- **Rationale**: Warmup values should not be artificially synthesized, forward-filled, or backward-filled. Generating estimates on partial windows ($< window$) inflates estimation variance and introduces spurious trading triggers. Downstream model pipelines are expected to drop or mask warmup rows.

### 3. Annualization Basis Factor
- **Chosen Convention**: Scaling by $\sqrt{252}$ as default parameter (`trading_days=252`).
- **Rationale**: Standard equities and equity derivatives trade approximately 252 business days per calendar year. The function supports overriding `trading_days=365` for continuous crypto/forex instruments.

