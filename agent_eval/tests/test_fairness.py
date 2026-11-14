"""
Fairness Tests for Agent Evaluation Graders.

Ensures prompt.md explicitly documents every public identifier, parameter,
file path, column name, and behavioral constraint evaluated by hidden tests.
Prevents secret parameters, hidden traps, or unstated evaluation criteria.
"""

from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def test_fairness_task_01_leakage():
    prompt = (TASKS_DIR / "task_01_leakage" / "prompt.md").read_text(encoding="utf-8")

    # Assert required files and function signatures are stated
    assert "feature_pipeline.py" in prompt
    assert "compute_features" in prompt
    assert "prepare_dataset" in prompt
    assert "window" in prompt
    assert "target_horizon" in prompt

    # Assert required feature columns are explicitly listed
    for col in ["returns", "rolling_mean", "rolling_std", "z_score", "volatility_ratio"]:
        assert col in prompt, f"Prompt must explicitly document column '{col}'"

    # Assert leakage and alignment rules are stated
    assert "center" in prompt or "centered" in prompt
    assert "bfill" in prompt or "backward" in prompt
    assert "target" in prompt


def test_fairness_task_02_ambiguous_spec():
    task_dir = TASKS_DIR / "task_02_ambiguous_spec"
    prompt = (task_dir / "prompt.md").read_text(encoding="utf-8")
    fairness_map = (task_dir / "FAIRNESS.md").read_text(encoding="utf-8")

    # Assert fairness map covers all key test functions
    for fn_name in [
        "test_public_api_signatures",
        "test_rolling_volatility_mathematical_precision",
        "test_annualized_scaling",
        "test_zero_variance_and_flat_prices",
        "test_invalid_inputs_and_short_series",
        "test_exact_window_plus_one_single_non_nan",
        "test_exponential_momentum_calculation",
        "test_notes_file_and_assumptions",
    ]:
        assert fn_name in fairness_map, f"FAIRNESS.md must map test '{fn_name}'"

    # Assert functions and parameters are documented
    assert "compute_rolling_volatility" in prompt
    assert "compute_exponential_momentum" in prompt
    assert "prices" in prompt
    assert "window" in prompt
    assert "annualized" in prompt
    assert "trading_days" in prompt
    assert "span" in prompt
    assert "indicators.py" in prompt

    # Assert documentation expectation is clearly stated
    assert "NOTES.md" in prompt
    assert "Technical Assumptions" in prompt
    assert "closed" in prompt or "right" in prompt
    assert "NaN" in prompt or "warmup" in prompt
    assert "252" in prompt or "annual" in prompt
    assert "<= window" in prompt
    assert "index" in prompt


def test_fairness_task_03_conventions():
    prompt = (TASKS_DIR / "task_03_conventions" / "prompt.md").read_text(encoding="utf-8")

    # Assert class and role names are documented
    assert "RiskStressAgent" in prompt
    assert "BaseAgent" in prompt
    assert "Stress Testing & Tail Risk" in prompt
    assert "step_index = 7" in prompt or "step_index: 7" in prompt or "7" in prompt

    # Assert Pydantic state model and fields are documented
    assert "StressReport" in prompt
    assert "stress_report" in prompt
    for field in ["scenario_name", "stressed_sharpe", "max_drawdown_stressed", "tail_risk_detected"]:
        assert field in prompt, f"Prompt must explicitly state StressReport field '{field}'"

    # Assert registry and file paths are documented
    assert "quantlab/orchestration/state.py" in prompt
    assert "quantlab/agents/stress_agent.py" in prompt
    assert "quantlab/agents/__init__.py" in prompt
    assert "quantlab/tools/registry.py" in prompt
    assert "run_stress_test" in prompt
    assert "quantlab/orchestration/engine.py" in prompt
    assert "tests/test_stress_agent.py" in prompt
    assert "docs/agents.md" in prompt


def test_fairness_task_04_nondeterminism():
    prompt = (TASKS_DIR / "task_04_nondeterminism" / "prompt.md").read_text(encoding="utf-8")

    # Assert function and file are documented
    assert "train_pipeline.py" in prompt
    assert "train_model" in prompt
    assert "seed" in prompt
    assert "epochs" in prompt

    # Assert all return dict keys are documented
    for key in ["model", "loss_history", "weight_hash", "final_loss", "feature_names"]:
        assert key in prompt, f"Prompt must explicitly state return key '{key}'"

    # Assert specific synchronization requirements are stated
    assert "random.seed" in prompt or "random" in prompt
    assert "np.random.seed" in prompt or "numpy" in prompt
    assert "torch.manual_seed" in prompt or "torch" in prompt
    assert "DataLoader" in prompt
    assert "sorted" in prompt or "alphabetical" in prompt
