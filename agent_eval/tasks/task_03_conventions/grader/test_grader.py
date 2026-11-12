"""Hidden pytest grader for Task 03: Repo Conventions & Multi-Agent Architecture."""

from pathlib import Path


def test_pydantic_state_schema():
    """Verify WorkflowState schema extended with typed StressReport Pydantic model."""
    from pydantic import BaseModel

    from quantlab.orchestration.state import StressReport, WorkflowState

    assert issubclass(StressReport, BaseModel), "StressReport must inherit from Pydantic BaseModel"

    fields = StressReport.model_fields
    expected_fields = ["scenario_name", "stressed_sharpe", "max_drawdown_stressed", "tail_risk_detected"]
    for f in expected_fields:
        assert f in fields, f"StressReport missing required field: {f}"

    state_fields = WorkflowState.model_fields
    assert "stress_report" in state_fields, "WorkflowState must contain 'stress_report' field"

    # Test instantiation & type validation
    rep = StressReport(
        scenario_name="2x_costs",
        stressed_sharpe=0.85,
        max_drawdown_stressed=-0.18,
        tail_risk_detected=False,
    )
    state = WorkflowState(stress_report=rep)
    assert state.stress_report.scenario_name == "2x_costs"


def test_agent_inherits_and_runs():
    """Verify RiskStressAgent inherits from BaseAgent and handles state transitions."""
    from quantlab.agents.stress_agent import RiskStressAgent

    from quantlab.agents.base import BaseAgent
    from quantlab.orchestration.state import BacktestResult, StressReport, WorkflowState

    assert issubclass(RiskStressAgent, BaseAgent), "RiskStressAgent must inherit from BaseAgent"

    agent = RiskStressAgent()
    assert agent.agent_name == "RiskStressAgent"
    assert agent.agent_role == "Stress Testing & Tail Risk"

    # Test execution with populated backtest result
    state = WorkflowState(backtest_result=BacktestResult(sharpe_ratio=1.5, max_drawdown=-0.10))
    updated_state = agent.run(state)

    assert isinstance(updated_state.stress_report, StressReport)
    assert len(updated_state.execution_trace) == 1
    trace = updated_state.execution_trace[0]
    assert trace.step_index == 7
    assert trace.agent_name == "RiskStressAgent"
    assert trace.status == "PASS"

    # Test edge case: state without backtest_result (must not crash)
    empty_state = WorkflowState()
    updated_empty = agent.run(empty_state)
    assert isinstance(updated_empty.stress_report, StressReport)


def test_agent_exported_in_all():
    """Verify RiskStressAgent is exported in quantlab.agents.__all__."""
    import quantlab.agents

    assert hasattr(quantlab.agents, "RiskStressAgent"), "RiskStressAgent must be exported from quantlab.agents"
    assert "RiskStressAgent" in quantlab.agents.__all__, "RiskStressAgent must be listed in quantlab.agents.__all__"


def test_registry_integration():
    """Verify tool_run_stress_test is registered in allowlisted ToolRegistry."""
    from quantlab.tools.registry import ToolRegistry

    registry = ToolRegistry()
    assert "run_stress_test" in registry.list_tools(), "Tool 'run_stress_test' not found in registry"
    tool_def = registry.get_tool("run_stress_test")
    assert tool_def is not None
    assert "scenario" in tool_def.args_schema.model_fields


def test_orchestrator_pipeline_sequence():
    """Verify MultiAgentOrchestrator executes 7 agents in sequence."""
    from quantlab.orchestration.engine import MultiAgentOrchestrator

    orchestrator = MultiAgentOrchestrator()
    assert len(orchestrator.agents) == 7, f"Expected 7 agents, found {len(orchestrator.agents)}"
    assert orchestrator.agents[-1].__class__.__name__ == "RiskStressAgent"

    final_state = orchestrator.run()
    assert len(final_state.execution_trace) == 7
    assert final_state.execution_trace[-1].agent_name == "RiskStressAgent"
    assert final_state.stress_report is not None


def test_agent_tests_exist_and_pass():
    """Verify test file tests/test_stress_agent.py exists and contains passing tests."""
    test_path = Path("tests/test_stress_agent.py")
    assert test_path.exists(), "tests/test_stress_agent.py must exist"

    # Import and run tests in the file
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_stress_agent", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find test functions and execute them
    test_funcs = [getattr(module, name) for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]
    assert len(test_funcs) >= 1, "tests/test_stress_agent.py must define at least one test_ function"
    for func in test_funcs:
        func()


def test_documentation_updated():
    """Verify docs/agents.md describes RiskStressAgent."""
    docs_path = Path("docs/agents.md")
    assert docs_path.exists(), "docs/agents.md must exist"
    content = docs_path.read_text(encoding="utf-8")
    assert "RiskStressAgent" in content, "docs/agents.md must document RiskStressAgent"
