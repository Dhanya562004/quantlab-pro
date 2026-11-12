# Task Prompt: Add a 7th Agent (RiskStressAgent) Following Repository Conventions

You are asked to extend QuantLab Pro's multi-agent quantitative pipeline by adding a 7th agent: `RiskStressAgent`.

To maintain architectural integrity, you must strictly follow all repository conventions across typing, registry, orchestrator, testing, and documentation.

## Requirements

### 1. Pydantic Typed State (`quantlab/orchestration/state.py`)
- Define a Pydantic model:
  ```python
  class StressReport(BaseModel):
      scenario_name: str
      stressed_sharpe: float
      max_drawdown_stressed: float
      tail_risk_detected: bool
  ```
- Add the typed field to `WorkflowState`:
  ```python
  stress_report: StressReport | None = Field(default=None)
  ```

### 2. Agent Implementation (`quantlab/agents/stress_agent.py`)
- Create `RiskStressAgent` inheriting from `BaseAgent` (`quantlab.agents.base.BaseAgent`).
- Set `agent_name = "RiskStressAgent"` and `agent_role = "Stress Testing & Tail Risk"`.
- Implement `run(self, state: WorkflowState) -> WorkflowState`:
  - Calculate or assign stress metrics:
    - Default scenario: `scenario_name = "2x_transaction_costs"`
    - If `state.backtest_result` is available, calculate `stressed_sharpe = state.backtest_result.sharpe_ratio * 0.8` and `max_drawdown_stressed = state.backtest_result.max_drawdown * 1.5`; otherwise use `stressed_sharpe = 0.0` and `max_drawdown_stressed = 0.0`.
    - Set `tail_risk_detected = bool(max_drawdown_stressed < -0.20 or stressed_sharpe < 0.0)`.
  - Instantiate `StressReport` and assign it to `state.stress_report`.
  - Append an `AgentStepTrace` to `state.execution_trace` with:
    - `step_index = 7`
    - `agent_name = "RiskStressAgent"`
    - `role = "Stress Testing & Tail Risk"`
    - `status = "PASS"`
    - `duration_ms` (float $> 0$)
    - `summary = "Stress testing completed."`
  - Return the updated `state`.

### 3. Agent Package Exports (`quantlab/agents/__init__.py`)
- Import `RiskStressAgent` in `quantlab/agents/__init__.py`.
- Include `"RiskStressAgent"` in `__all__`.

### 4. Allowlisted Tool Registry (`quantlab/tools/registry.py`)
- Define an argument model:
  ```python
  class StressTestArgs(BaseModel):
      scenario: str = "2x_transaction_costs"
  ```
- Define and register the tool function `tool_run_stress_test` in `ToolRegistry` with name `"run_stress_test"`.

### 5. Multi-Agent Orchestrator Pipeline (`quantlab/orchestration/engine.py`)
- In `MultiAgentOrchestrator.__init__`, append `RiskStressAgent()` to `self.agents` so that the pipeline runs all 7 agents in sequence.

### 6. Unit Testing (`tests/test_stress_agent.py`)
- Create `tests/test_stress_agent.py` containing a pytest test function that verifies `RiskStressAgent().run(state)` produces a valid `state.stress_report` and appends an `AgentStepTrace` to `state.execution_trace`.

### 7. Documentation (`docs/agents.md`)
- Update `docs/agents.md` to describe `RiskStressAgent` as Agent 7 in the multi-agent pipeline.

