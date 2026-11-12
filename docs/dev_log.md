# QuantLab Pro — Development Log & Agentic Coding Evidence

## Overview
This development log documents verifiable engineering tasks executed during the QuantLab Pro quantitative ML research platform upgrade.

---

## Log Entry: 2026-10-10 (Full Platform Architecture Upgrade)

- **Date & Time:** `2026-10-10 09:35:00 UTC`
- **Assistant / Tool Used:** `Antigravity AI Assistant (Gemini 3.6 Flash)`
- **Task / Prompt:** Upgrade QuantLab Pro into a fully tested, documented, reproducible quantitative ML platform meeting Tower Research Capital's AI/ML internship requirements across 9 core sections.

### Files Modified & Created
1. `quantlab/agents/__init__.py`, `planner_agent.py`, `training_agent.py`, `evaluation_agent.py`, `auditor_agent.py` — Exported exact class aliases (`PlannerAgent`, `TrainingAgent`, `EvaluationAgent`, `AuditorAgent`) and verified 6-stage sequential execution.
2. `mcp_server/server.py` & `quantlab/tools/domain_tools.py` — Registered allowlisted domain tools (`inspect_dataset`, `dataset_summary`, `validate_dataset`, `run_experiment`, `get_experiment_metrics`, `get_experiment`, `get_model_metrics`, `get_backtest_summary`).
3. `mcp_server/client.py` & `tests/test_mcp_server.py` — Implemented genuine stdio transport integration test using official Python MCP SDK (`ClientSession`, `stdio_client`, `StdioServerParameters`).
4. `quantlab/features/builder.py` — Fixed `valid_mask` target shifting to drop NaNs from `future_return`, ensuring leakage-safe feature construction.
5. `quantlab/evaluation/backtest.py` — Enhanced backtest calculation robustness with `np.nan_to_num` and early input safety checks.
6. `quantlab/models/trainer.py` & `tests/test_models_and_metrics.py` — Implemented `save_model_artifact` and `load_model_artifact` with PyTorch weight serialization and metadata manifest link.
7. `api/main.py` & `tests/test_api.py` — Added `/ready` endpoint verifying SQLite database readiness and expanded REST test coverage.
8. `tests/test_backtest.py` — Added dedicated unit tests for transaction costs, turnover, drawdown, Sharpe/Sortino edge cases, and zero volatility.
9. `README.md` & `docs/resume_evidence.md` — Updated documentation with Mermaid architecture diagram, model evaluation table, MCP server startup guide, and resume evidence.

### Commands & Tests Executed
```bash
pytest -v
ruff check . --fix
python -m mcp_server.client
```

### Observed Results
- **PyTest Suite:** `39 passed in 18.61s` (0 failed, 0 skipped, 0 errors).
- **Ruff Linter:** `All checks passed!` (0 errors).
- **MCP SDK Stdio Integration Test:** Clean connection, tool discovery (7 tools), execution of `dataset_summary` and `run_experiment`, invalid input rejection, and clean process termination.

### Remaining Limitations
1. Live market data via `yfinance` remains optional; offline synthetic dataset generation guarantees 100% offline reproducibility without third-party rate limits.
2. Production REST server deployment requires process management (e.g. Uvicorn/Gunicorn daemon with systemd or Docker containerization).

---

## Reusable Assistant Prompt Template

The following prompt template can be used with AI coding assistants (e.g. Antigravity, Claude Code, Cursor) to reproduce repository audits and upgrades safely:

```markdown
Act as a senior Quantitative ML Engineer and MLOps Engineer.
1. Inspect the repository structure, codebase, and test suite.
2. Verify all test files under `tests/` using `pytest -v` and check code quality using `ruff check .`.
3. Implement missing agent orchestration steps, state validation, and model artifact persistence.
4. Execute the three-model benchmark (Majority baseline, Logistic Regression, PyTorch MLP) on held-out test splits.
5. Verify stdio MCP server-client tool invocation using the official MCP Python SDK.
6. Update documentation with verified test counts and exact empirical metrics without using placeholders or unverified data.
```

---

## Log Entry: 2026-11-12 (RL-Task Builder Agent Evaluation Harness)

- **Date & Time:** `2026-11-12 14:30:00 UTC`
- **Assistant / Tool Used:** `Antigravity AI Assistant (Gemini 3.8 Flash)`
- **Task / Prompt:** Build an RL-task style agent evaluation subproject (`agent_eval/`) featuring 4 hard tasks with at least two separate sources of difficulty, automated procedural pytest graders, rubric checks, determinism verification, import and LLM agent runners, empirical reporting from real runs only, CI integration, and documentation updates.

### Files Modified & Created
1. `agent_eval/runner/`:
   - `agents.py`: Defined `AgentRunner` abstract base class, `ImportAgentRunner` for loading pre-existing solutions from disk/diffs, `DirectoryOverlayRunner` for local overlays, and `LLMAgentRunner` supporting Gemini, OpenAI, and Anthropic APIs.
   - `run_task.py`: Isolated evaluation runner copying starters to fresh temporary directories, applying agent solutions, executing hidden graders under subprocess timeouts, evaluating rubrics, and saving run JSON manifests.
   - `check_determinism.py`: Runs task graders $N$ times (default 20, or $N=5$ in CI) to ensure zero flakiness.
   - `report.py`: Aggregates runs from `agent_eval/runs/` into an empirical pass-rate table without fabricating numbers.
2. `agent_eval/tasks/`:
   - `task_01_leakage/`: Look-ahead leakage in rolling normalization (`center=True`) and backward fill (`bfill`), with shifted-target future perturbation tests and public API invariance checks. Includes starter, hidden grader, rubric, reference solution, and 2 known-bad solutions.
   - `task_02_ambiguous_spec/`: Realized volatility and exponential momentum with boundary ambiguity (`closed='right'`), Bessel's correction (`ddof=1`), flat price handling, and explicit assumption documentation required in `NOTES.md`. Includes starter, hidden grader, rubric, reference solution, and 2 known-bad solutions.
   - `task_03_conventions/`: Adding a 7th agent (`RiskStressAgent`) adhering to repo conventions: typed Pydantic `StressReport` state, registration in `ToolRegistry`, orchestrator sequencing, unit tests, and docs sync. Includes starter, hidden grader, rubric, reference solution, and 2 known-bad solutions.
   - `task_04_nondeterminism/`: Multi-library RNG synchronization, unordered `set()` iteration sorting, and DataLoader generator pinning for PyTorch bit-for-bit weight hash determinism. Includes starter, hidden grader, rubric, reference solution, and 2 known-bad solutions.
3. `agent_eval/tests/`:
   - `test_reference_solutions.py`: Asserts all 4 reference solutions pass 100% of hidden grader tests.
   - `test_known_bads.py`: Asserts all 8 known-bad solutions fail the grader.
   - `test_fairness.py`: Asserts that `prompt.md` explicitly documents every public identifier, signature, parameter, and file path evaluated by hidden tests.
   - `test_determinism.py`: Asserts that all task graders produce 100% identical outcomes over $N=5$ iterations.
4. `agent_eval/FAILURE_ANALYSIS.md`: Failure analysis template with one section per task marked strictly with `TODO: fill from runs/`.
5. `agent_eval/README.md`: Harness architecture, execution guide, grading rules, and limitations.
6. `README.md`: Root README updated to include the Agent Evaluation Harness section, updated project positioning, and removed resume evidence section.
7. `.github/workflows/ci.yml`: Added step executing grader self-tests alongside the original test suite.
8. `pyproject.toml`: Added `pythonpath = ["."]` to `tool.pytest.ini_options`.

### Commands & Tests Executed
```bash
pytest tests/ -v
pytest agent_eval/tests/ -v
python -m agent_eval.runner.check_determinism -n 5
python -m agent_eval.runner.report
ruff check .
```

### Observed Results
- **Original Test Suite:** `39 passed in 73.63s` (100% pass rate preserved).
- **Grader Self-Tests:** `20 passed in 130.55s` (4 reference solutions passed, 8 known bads failed, 4 fairness tests passed, 4 determinism tests passed).
- **Determinism Checker:** All 4 tasks confirmed 100% deterministic over 5 repetitions with zero flakiness.
- **Report Script:** Accurately reported 0 runs and displayed clear TODO placeholders without fabricated data.
- **Ruff Linter:** `All checks passed!` (0 lint errors across entire repository).

### Mistakes Made & Caught During Development
1. **Task 1 Warmup Perturbation Indexing:** In `test_no_backward_fill_leakage`, the initial assertion tested modifying row 25, which lies after the initial 20-bar window. Because `bfill()` was propagating row 19 backward into rows 0..18, perturbing row 25 did not affect row 0, causing `bad_01_only_center_fixed` to unexpectedly pass. This was caught during grader self-tests (`test_known_bads.py`). The test was updated to verify that row 0 `rolling_mean` is strictly NaN and that modifying row 19 does not contaminate row 0, successfully catching the `bfill` leakage.
2. **PyTest Import Path for agent_eval:** Initial standalone execution of `pytest agent_eval/tests` produced `ModuleNotFoundError: No module named 'agent_eval'` due to `testpaths = ["tests"]` in `pyproject.toml`. Added `agent_eval/__init__.py` and configured `pythonpath = ["."]` in `[tool.pytest.ini_options]`, allowing direct test execution without manual path overrides.
3. **Ruff Import Formatting:** Initial file generation had minor unused import warnings and unsorted import blocks. Ran `ruff check --fix .` and verified `All checks passed!`.
