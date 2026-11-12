"""Unit tests for RiskStressAgent."""

from quantlab.agents.stress_agent import RiskStressAgent

from quantlab.orchestration.state import BacktestResult, WorkflowState


def test_risk_stress_agent_execution():
    agent = RiskStressAgent()
    state = WorkflowState(backtest_result=BacktestResult(sharpe_ratio=2.0, max_drawdown=-0.08))
    updated = agent.run(state)

    assert updated.stress_report is not None
    assert updated.stress_report.scenario_name == "2x_transaction_costs"
    assert updated.stress_report.stressed_sharpe == 1.6
    assert len(updated.execution_trace) == 1
    assert updated.execution_trace[0].step_index == 7
    assert updated.execution_trace[0].status == "PASS"
