"""Known-bad solution 1: Omits registry registration and __all__ export."""

from quantlab.agents.base import BaseAgent
from quantlab.orchestration.state import AgentStepTrace, StressReport, WorkflowState


class RiskStressAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskStressAgent", "Stress Testing & Tail Risk")

    def run(self, state: WorkflowState) -> WorkflowState:
        state.stress_report = StressReport(
            scenario_name="2x_transaction_costs",
            stressed_sharpe=0.5,
            max_drawdown_stressed=-0.1,
            tail_risk_detected=False,
        )
        state.execution_trace.append(
            AgentStepTrace(
                step_index=7,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="PASS",
                duration_ms=1.0,
                summary="Stress testing done.",
            )
        )
        return state
