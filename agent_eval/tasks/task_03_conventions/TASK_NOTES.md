# Task 03: Repository Architecture & Multi-Agent Convention Adherence

## Overview
This task evaluates how well an agent discovers and complies with non-trivial repository conventions across a multi-component architecture. Instead of editing an isolated script, the agent must add a new 7th agent (`RiskStressAgent`) to an existing multi-agent pipeline while satisfying all existing architectural conventions.

## Sources of Difficulty (Why This Task Is Hard)

### 1. Multi-File Convention Coordination
In enterprise repositories, a feature addition is rarely confined to one file. The agent must discover and update 6 distinct layers:
1. **Type Layer (`quantlab/orchestration/state.py`):** Define a Pydantic `StressReport` model and add `stress_report: StressReport | None` to the central `WorkflowState`.
2. **Agent Layer (`quantlab/agents/stress_agent.py`):** Inherit from `BaseAgent`, implement `run(self, state: WorkflowState) -> WorkflowState`, conform to Sphinx/Google docstring standards, and append an `AgentStepTrace` with `step_index=7`.
3. **Module Export Layer (`quantlab/agents/__init__.py`):** Export `RiskStressAgent` and update `__all__`.
4. **Tool Registry Layer (`quantlab/tools/registry.py`):** Register the agent's capability as a typed tool (`tool_run_stress_test`) in the allowlisted `ToolRegistry` with a Pydantic arguments schema.
5. **Orchestrator Layer (`quantlab/orchestration/engine.py`):** Add the agent as the 7th element in `MultiAgentOrchestrator.agents`.
6. **Documentation & Testing Layer (`docs/agents.md` & `tests/test_stress_agent.py`):** Write unit tests covering state transitions and document the 7-agent topology.

Coding agents typically perform a subset of these tasks (e.g. creating the agent class) while forgetting registry registration, typed state definitions, unit tests, or documentation updates.

### 2. Pydantic Type Enforcement & Edge Case Resilience
The state transition must strictly enforce typed Pydantic validation:
- Assigning a raw Python `dict` or untyped object to `state.stress_report` fails schema validation.
- The agent must be resilient to partial workflow runs (e.g., if previous agent steps like backtesting were skipped or yielded `None`). Crashing with an `AttributeError` on `state.backtest_result` fails robustness tests.

