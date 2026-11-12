# Task 02: Ambiguous Specification & Boundary Convention Handling

## Overview
This task tests how an agent handles specification ambiguity in quantitative financial indicators. In real production quant environments, engineering tickets frequently omit subtle edge-case conventions (such as rolling window boundary closures, warmup policies, Bessel's correction, and annualization bases). High-performing agents recognize ambiguity, adopt standardized idioms, and document their assumptions explicitly in a `NOTES.md` file.

## Sources of Difficulty (Why This Task Is Hard)

### 1. Mathematical Rigor & Extreme Edge Cases
Implementing rolling financial metrics seems simple, but standard implementations break down under production edge cases:
- **Zero Variance / Flat Markets:** When an asset halts or prices remain constant across a window, standard deviation is zero. Naive implementations that divide by standard deviation or compute percentage changes without checking for flat returns produce `NaN` or unhandled zero-division warnings.
- **Short Series (< Window):** Calling code may pass short or fragmented data series. The function must return a series of identical length and index filled with `NaN`s rather than raising an unhandled `IndexError` or crashing.
- **Bessel's Correction (`ddof=1`):** Sample variance must use $N-1$ degrees of freedom (`ddof=1`). Agents that default to numpy (`ddof=0`) fail precision checks against analytical targets.
- **Non-positive Prices:** Exponential momentum divides by exponential moving average (`(prices - ema) / ema`). If prices contain non-positive values, it must validate and raise a descriptive `ValueError`.

### 2. Specification Ambiguity & Explicit Assumption Documentation
In quantitative time-series modeling:
- **Window Closure:** `closed='right'` (including the current bar $t$) vs. `closed='left'` (using strictly prior bars $t-1$).
- **Warmup Policy:** Should warmup periods be interpolated, filled, or preserved as `NaN`s? (Preserving `NaN`s avoids spurious signals).
- **Annualization Factor:** Defaulting to 252 (equities) while allowing configurable `trading_days`.
The prompt explicitly directs the agent to document all architectural decisions in `NOTES.md` under `## Technical Assumptions`. Coding agents frequently ignore documentation requirements when writing code, resulting in grader failures.

