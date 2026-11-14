# Technical Architecture & Convention Notes

## Technical Assumptions

### 1. Rolling Window Boundary Closure
- **Chosen Convention**: `closed='right'`.

### 2. Warmup NaN Policy
- **Chosen Convention**: Preserve `NaN`s for initial $window - 1$ bars, producing first valid value at index $window - 1$.
- **Rationale**: Assumes $window$ price bars allow computing rolling return volatility immediately at bar $window - 1$.

### 3. Annualization Basis Factor
- **Chosen Convention**: Scaling by $\sqrt{252}$ as default parameter (`trading_days=252`).

