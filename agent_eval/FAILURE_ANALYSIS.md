# Failure Analysis Log

This document records empirical findings from 20 valid real agent execution runs in `agent_eval/runs/`.

## Evaluation Setup

- **Agent:** LLM agent
- **Model:** `gemini-3.5-flash-lite`
- **Mode:** `oneshot`
- **Total valid runs:** 20 (5 per task)
- **Overall result:** 14 passed, 6 failed (70% pass rate)
- **Excluded records:** `pilot_run_01` was a preliminary pilot. `task01_run_05` was a harness error caused by malformed Markdown and was excluded from the 20 valid runs.

---

## Task 01: Look-Ahead Data Leakage (`task_01_leakage`)

- **Agent Evaluated:** `llm_agent`, using `gemini-3.5-flash-lite`.
- **What the Agent Did:** The agent modified `feature_pipeline.py` to calculate returns, rolling statistics, z-scores, and volatility ratios. It used non-centered rolling windows and did not use backward filling.
- **Where It Failed:** All 5 valid runs failed 3 dataset alignment tests. The `prepare_dataset` implementation used `shift(-target_horizon)` to create future targets and then called `dropna()`, but it did not correctly remove the final rows with missing future targets. The actual output retained rows beyond the expected final index.
- **Root Cause Classification (Model Gap vs. Grader Mistake):** Model implementation gap. The grader consistently detected incorrect target alignment. The other 5 checks passed, including feature calculations and leakage-related checks.
- **Remediation & Grader Adjustments:** Correct the dataset alignment logic so rows without valid future targets are removed. Retain the failing alignment tests because they identify a real implementation defect. No grader change is justified by the observed evidence.

**Result:** 0/5 runs passed.

---

## Task 02: Ambiguous Specification & Boundary Conventions (`task_02_ambiguous_spec`)

- **Agent Evaluated:** `llm_agent`, using `gemini-