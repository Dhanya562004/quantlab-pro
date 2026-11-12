# Rubric: Task 02 - Ambiguous Spec

## Procedural Test Checks
1. `test_public_api_signatures`: Signature parameters and defaults for `compute_rolling_volatility` and `compute_exponential_momentum`.
2. `test_rolling_volatility_mathematical_precision`: Sample standard deviation with Bessel's correction (`ddof=1`) matching analytical return volatility.
3. `test_annualized_scaling`: Scaling factors $\sqrt{252}$ and $\sqrt{365}$ accurately applied.
4. `test_zero_variance_and_flat_prices`: Flat prices return 0.0 without errors or NaNs post-warmup.
5. `test_invalid_inputs_and_short_series`: Handles series shorter than `window` by returning all NaNs; rejects non-positive prices with `ValueError`.
6. `test_exponential_momentum_calculation`: Verified against analytical EMA formula.
7. `test_notes_file_and_assumptions`: Validates `NOTES.md` file exists and documents window closure, warmup NaNs, and annualization under `## Technical Assumptions`.

## Rubric Checks
- `NOTES.md` exists and contains substantive explanation (>20 characters) documenting technical trade-offs.

