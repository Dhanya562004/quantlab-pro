"""
Unit & Integration Tests for Multi-Agent Orchestration Workflow (QuantLab Pro).
Verifies sequential step trace, failure halting, and state passing across all 6 agents.
"""


from quantlab.data.synthetic import generate_synthetic_ohlcv
from quantlab.orchestration.engine import MultiAgentOrchestrator


def test_end_to_end_multi_agent_pipeline():
    df, meta = generate_synthetic_ohlcv(symbol="AGENT_TEST", n_bars=300, seed=42)
    orchestrator = MultiAgentOrchestrator()

    state = orchestrator.run_pipeline(
        df_raw=df,
        raw_metadata=meta,
        model_choice="logistic_regression",
        seed=42,
    )

    assert state.is_completed is True
    assert state.error_message is None
    assert len(state.execution_trace) == 6

    # Check trace step sequence
    step_names = [step.agent_name for step in state.execution_trace]
    assert step_names == [
        "Data Quality Agent",
        "Quant Research Planner Agent",
        "Feature Engineering Agent",
        "Model Training Agent",
        "Evaluation & Risk Agent",
        "Experiment Auditor Agent",
    ]

    # Verify outputs populated in state
    assert state.validation_report is not None
    assert state.validation_report.is_valid is True
    assert state.feature_config is not None
    assert state.test_metrics is not None
    assert state.backtest_result is not None
    assert state.experiment_id.startswith("EXP-")
    assert state.is_persisted is True


def test_agent_pipeline_failure_halting():
    df, meta = generate_synthetic_ohlcv(symbol="FAIL_TEST", n_bars=30, seed=42)
    orchestrator = MultiAgentOrchestrator()

    # Will fail in Data Quality Agent due to small sample size (n=30 < 100)
    state = orchestrator.run_pipeline(
        df_raw=df,
        raw_metadata=meta,
        model_choice="logistic_regression",
        seed=42,
    )

    assert state.is_completed is False
    assert state.error_message is not None
    assert "Data Quality Validation Failed" in state.error_message
    assert state.execution_trace[0].status == "FAIL"


def test_agent_aliases_and_execution_order():
    from quantlab.agents import (
        AuditorAgent,
        DataQualityAgent,
        EvaluationAgent,
        FeatureEngineeringAgent,
        PlannerAgent,
        TrainingAgent,
    )
    agents = [
        DataQualityAgent(),
        PlannerAgent(),
        FeatureEngineeringAgent(),
        TrainingAgent(),
        EvaluationAgent(),
        AuditorAgent(),
    ]
    assert len(agents) == 6
    assert agents[0].agent_name == "Data Quality Agent"
    assert agents[1].agent_name == "Quant Research Planner Agent"
    assert agents[2].agent_name == "Feature Engineering Agent"
    assert agents[3].agent_name == "Model Training Agent"
    assert agents[4].agent_name == "Evaluation & Risk Agent"
    assert agents[5].agent_name == "Experiment Auditor Agent"


def test_invalid_pydantic_state_and_no_false_success():
    from quantlab.orchestration.state import WorkflowState
    df, meta = generate_synthetic_ohlcv(symbol="VALID_TEST", n_bars=250, seed=42)
    orchestrator = MultiAgentOrchestrator()

    # Create invalid state missing df_raw
    state = WorkflowState(df_raw=None, raw_metadata=meta)

    # First agent (DataQualityAgent) should fail and halt pipeline
    res_state = orchestrator.agents[0].run(state)
    assert res_state.is_completed is False
    assert res_state.error_message is not None
    assert "No dataset provided" in res_state.error_message
    assert res_state.execution_trace[-1].status == "FAIL"


def test_recoverable_failure_and_controlled_termination():
    df, meta = generate_synthetic_ohlcv(symbol="RECOVER_TEST", n_bars=200, seed=42)
    orchestrator = MultiAgentOrchestrator()

    # Invalid model choice gracefully falls back to default in PlannerAgent
    state = orchestrator.run_pipeline(
        df_raw=df,
        raw_metadata=meta,
        model_choice="unknown_model_type_xyz",
        seed=42,
    )

    assert state.is_completed is True
    assert state.selected_model_type == "logistic_regression"
    assert state.execution_trace[1].status == "PASS"

