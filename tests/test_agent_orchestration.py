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
