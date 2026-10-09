"""
Typed Workflow State & Step Trace Representation for QuantLab Pro.
"""

from typing import Any

from pydantic import BaseModel, Field

from quantlab.data.validation import ValidationReport
from quantlab.evaluation.backtest import BacktestResult
from quantlab.evaluation.metrics import ClassificationMetrics
from quantlab.features.builder import DatasetSplits, FeatureConfig
from quantlab.models.trainer import TrainedModelArtifact


class AgentStepTrace(BaseModel):
    """Execution log trace for a single agent step."""
    step_index: int
    agent_name: str
    role: str
    status: str = Field(description="PASS, FAIL, or SKIPPED")
    duration_ms: float
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """
    Central typed state passed between agents in the deterministic orchestration workflow.
    """
    # Inputs
    df_raw: Any | None = Field(default=None, exclude=True)
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    user_model_choice: str = Field(default="logistic_regression")
    user_seed: int = Field(default=42)
    user_target_horizon: int = Field(default=1)
    user_transaction_cost_bps: float = Field(default=10.0)

    # Agent 1 Output: Data Quality
    validation_report: ValidationReport | None = None

    # Agent 2 Output: Research Plan
    feature_config: FeatureConfig | None = None
    selected_model_type: str = Field(default="logistic_regression")
    research_rationale: str = Field(default="")

    # Agent 3 Output: Feature Engineering & Splits
    splits: DatasetSplits | None = Field(default=None, exclude=True)
    aligned_sample_count: int = Field(default=0)
    feature_names: list[str] = Field(default_factory=list)

    # Agent 4 Output: Model Training
    trained_artifact: TrainedModelArtifact | None = None
    trained_model_object: Any | None = Field(default=None, exclude=True)

    # Agent 5 Output: Evaluation & Risk
    test_metrics: ClassificationMetrics | None = None
    backtest_result: BacktestResult | None = None
    risk_warnings: list[str] = Field(default_factory=list)

    # Agent 6 Output: Audit & Manifest
    experiment_id: str = Field(default="")
    manifest: dict[str, Any] = Field(default_factory=dict)
    is_persisted: bool = Field(default=False)

    # Trace log
    execution_trace: list[AgentStepTrace] = Field(default_factory=list)
    is_completed: bool = Field(default=False)
    error_message: str | None = None

    model_config = {"arbitrary_types_allowed": True}
