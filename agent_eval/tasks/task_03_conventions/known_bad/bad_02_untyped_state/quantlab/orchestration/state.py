"""Known-bad solution 2: Uses untyped dictionary in state instead of Pydantic model."""

from typing import Any

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    sharpe_ratio: float = 1.2
    max_drawdown: float = -0.15

class AgentStepTrace(BaseModel):
    step_index: int
    agent_name: str
    role: str
    status: str
    duration_ms: float
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)

# FLAW: StressReport is not defined as a BaseModel, and stress_report is an untyped dict
class WorkflowState(BaseModel):
    raw_data: Any | None = None
    backtest_result: BacktestResult | None = None
    stress_report: dict[str, Any] | None = Field(default=None)
    execution_trace: list[AgentStepTrace] = Field(default_factory=list)
    is_completed: bool = False
