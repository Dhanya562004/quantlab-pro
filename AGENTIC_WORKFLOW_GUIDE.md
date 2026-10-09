# Agentic Coding Tool Workflow Guide

This guide details developer workflows for inspecting, modifying, refactoring, and extending **QuantLab Pro** using AI coding assistants such as **Claude Code**, **Cursor**, and **Codex**.

---

## 1. Environment & Architecture Overview

QuantLab Pro is structured as a modular quantitative ML research platform:
- `quantlab/data/`: Synthetic generator, CSV parser, yfinance loader, quantitative data validator.
- `quantlab/features/`: Leakage-resistant technical feature engineering and chronological train/val/test splits.
- `quantlab/models/`: Majority Class baseline, Logistic Regression, and PyTorch MLP classifier.
- `quantlab/agents/`: 6 specialized quantitative research agents (`DataQualityAgent`, `QuantResearchPlannerAgent`, `FeatureEngineeringAgent`, `ModelTrainingAgent`, `EvaluationRiskAgent`, `ExperimentAuditorAgent`).
- `quantlab/orchestration/`: Multi-Agent Orchestrator engine and typed `WorkflowState`.
- `quantlab/tools/`: Allowlisted safe tool registry with Pydantic argument schemas.
- `quantlab/evaluation/`: Held-out classification metrics and research-only backtesting engine.
- `quantlab/storage/`: SQLite database storage (`ExperimentStorage`).
- `mcp_server/`: Official Model Context Protocol (MCP) server & client.
- `api/`: FastAPI REST inference service.

---

## 2. Sample Agentic Coding Prompt Workflows

### Scenario A: Refactoring Feature Engineering (Preventing Data Leakage)
**Prompt for Claude Code / Cursor**:
> "Inspect `quantlab/features/builder.py`. Verify that all rolling windows (RSI, MACD, SMA) use only historical observations up to time `t`. Add a new technical feature `vol_ratio_20` representing the ratio of 5-day rolling volatility to 20-day rolling volatility. Update unit tests in `tests/test_leakage_and_splits.py` to assert that `vol_ratio_20` is present in `X` and run `pytest -v`."

### Scenario B: Adding a New ML Model Architecture
**Prompt for Claude Code / Cursor**:
> "Create a new model wrapper `RandomForestBaseline` in `quantlab/models/baselines.py` using `sklearn.ensemble.RandomForestClassifier`. Update `quantlab/models/trainer.py` to support `model_type='random_forest'`. Add unit tests in `tests/test_models_and_metrics.py` and verify all tests pass."

### Scenario C: Debugging Data Validation Failures
**Prompt for Claude Code / Cursor**:
> "Run `pytest tests/test_data_validation.py`. If any assertion fails, inspect `quantlab/data/validation.py`, print the exact validation audit trail log, fix the boundary condition check, and confirm `pytest` passes cleanly."

---

## 3. Quantitative Code Review Checklist

When reviewing code diffs produced by agentic coding tools, enforce these strict quantitative standards:

- [ ] **Zero Look-Ahead Leakage**: Are technical features calculated strictly using data at or before period `t`?
- [ ] **Chronological Split Boundaries**: Is data split chronologically without random shuffling?
- [ ] **Scaler Scope**: Is `StandardScaler` (or any normalization transformer) fit **only** on the training split (`X_train`)?
- [ ] **Target Alignment**: Is the target `y_t` correctly shifted to represent return from `t` to `t+h`?
- [ ] **Deterministic Seed Control**: Are random seeds explicitly passed to NumPy (`np.random.RandomState`), PyTorch (`torch.manual_seed`), and scikit-learn models?
- [ ] **Transaction Costs**: Does the backtesting engine deduct transaction costs (`bps / 10000.0`) on position transitions?
- [ ] **Input Validation**: Are tool input arguments validated via Pydantic schemas before execution?
- [ ] **Test Coverage**: Does every new feature have corresponding unit tests in `tests/`?
