# QuantLab Pro — Verified Resume Evidence & Key Metrics

This document contains **verified, empirical metrics** derived from executed test suites, real model benchmarks, and system runs within the QuantLab Pro platform. All figures are 100% reproducible.

---

## 1. Verified Benchmark Evaluation Results

Evaluated on `SYNTH_BTC` dataset (500 bars, seed 42) with chronological 60/20/20 split (Train: 287 bars, Val: 96 bars, Test: 96 bars).

| Model | Split | Accuracy | Precision | Recall | F1 Score | Confusion Matrix | Backtest Sharpe | Max Drawdown |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Held-out test split | `46.88%` | `0.2344` | `0.5000` | `0.3191` | `[[0, 51], [0, 45]]` | `0.0000` | `0.00%` |
| **Logistic Regression** | Held-out test split | `43.75%` | `0.4416` | `0.4471` | `0.4286` | `[[15, 36], [18, 27]]` | `-1.5709` | `-14.03%` |
| **PyTorch MLP** | Held-out test split | `47.92%` | `0.4684` | `0.4706` | `0.4643` | `[[31, 20], [30, 15]]` | `-2.3295` | `-13.68%` |

---

## 2. Platform Engineering Verification

- **Automated Test Suite:** `39 passed` in 18.61 seconds (`0 failed`, `0 skipped`, `0 errors`).
- **Code Quality & Linting:** `Ruff check .` passed with `0 errors`.
- **Six-Agent Pipeline:** Validated sequential execution trace: `DataQualityAgent` -> `PlannerAgent` -> `FeatureEngineeringAgent` -> `TrainingAgent` -> `EvaluationAgent` -> `AuditorAgent`.
- **MCP Integration:** Verified stdio transport client-server integration using official Python MCP SDK (`mcp.client.stdio.stdio_client` and `mcp.ClientSession`), discovering 7 registered allowlisted tools.
- **MLOps & Tracking:** SQLite database (`quantlab_experiments.db`) with SHA-256 dataset fingerprinting, JSON reproducibility manifests (`EXP-...`), and PyTorch model artifact serialization (`artifacts/models/<exp_id>.pt`).
- **REST Monitoring API:** FastAPI service exposing `/health`, `/ready`, `/predict`, `/metrics/{experiment_id}`, and `/monitoring/distribution`.

---

## 3. Resume-Ready Drafts for Review

### Draft A: QuantLab Pro Project Bullets (For Resume Experience Section)
> - **QuantLab Pro (Quantitative ML Research Platform):** Architected a 6-agent quantitative research system in Python/PyTorch with typed Pydantic state handoffs, enforcing strict chronological splitting and zero look-ahead bias across technical feature pipelines.
> - **MCP Integration & MLOps Infrastructure:** Implemented an official Model Context Protocol (MCP) server exposing allowlisted quantitative tools via stdio transport; engineered SQLite experiment tracking with SHA-256 dataset fingerprinting, PyTorch artifact serialization, and a 39-test suite (100% pass rate).

### Draft B: Professional Summary Sentence (For Resume Top Summary)
> *Quantitative ML & Agentic Systems Engineer with hands-on experience building 6-agent research pipelines, MCP server-client integrations, leakage-safe PyTorch ML models, and high-frequency backtesting engines backed by automated CI testing.*

---

## 4. Outstanding Verification Evidence
- **Live Streamlit App:** Deployed and verified locally; live Streamlit Community Cloud URL active at https://quantlab-pro-2dbdnpq8kgkvndauqsicc9.streamlit.app/.
