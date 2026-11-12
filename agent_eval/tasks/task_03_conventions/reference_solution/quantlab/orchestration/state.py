"""Typed Workflow State Representation."""

from typing import Any

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    sharpe_ratio: float = 1.2
    max_drawdown: float = -0.15


class AgentStepTrace(BaseModel):
    step_index: int
    agent_name: str
    role: str
    status: str = Field(description="PASS, FAIL, or SKIPPED")
    duration_ms: float
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class StressReport(BaseModel):
    """Stress test and tail risk assessment metrics."""
    scenario_name: str
    stressed_sharpe: float
    max_drawdown_stressed: float
    tail_risk_detected: bool


class WorkflowState(BaseModel):
    """Workflow state passed through multi-agent sequence."""
    raw_data: Any | None = None
    backtest_result: BacktestResult | None = None
    stress_report: StressReport | None = Field(default=None)
    execution_trace: list[AgentStepTrace] = Field(default_factory=list)
    is_completed: bool = False

    model_config = {"arbitrary_types_allowed": True}
