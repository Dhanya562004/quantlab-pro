"""Tests for multi-agent orchestrator."""

from quantlab.orchestration.engine import MultiAgentOrchestrator


def test_orchestrator_initial_run():
    orchestrator = MultiAgentOrchestrator()
    assert len(orchestrator.agents) >= 6
    state = orchestrator.run()
    assert state.is_completed is True
    assert len(state.execution_trace) >= 6

