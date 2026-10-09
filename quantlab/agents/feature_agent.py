"""
Feature Engineering Agent for QuantLab Pro.
Constructs leakage-resistant technical features, target alignment, and chronological splits.
"""

import time

from quantlab.agents.base import BaseAgent
from quantlab.features.builder import (
    build_features_and_target,
    create_chronological_splits,
)
from quantlab.orchestration.state import AgentStepTrace, WorkflowState


class FeatureEngineeringAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Feature Engineering Agent",
            agent_role="Constructs leakage-safe features, shifts target, and fits StandardScaler on train split only."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()

        if state.feature_config is None or state.df_raw is None:
            duration = (time.time() - start_t) * 1000
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="SKIPPED",
                duration_ms=round(duration, 2),
                summary="Skipped due to missing feature config or dataset.",
            ))
            return state

        try:
            X, y, feature_cols, df_aligned = build_features_and_target(state.df_raw, state.feature_config)
            splits = create_chronological_splits(X, y, df_aligned, state.feature_config)

            state.splits = splits
            state.aligned_sample_count = len(df_aligned)
            state.feature_names = feature_cols

            duration = (time.time() - start_t) * 1000

            summary = (
                f"Generated {len(feature_cols)} technical features on {len(df_aligned)} aligned bars. "
                f"Train={len(splits.X_train)}, Val={len(splits.X_val)}, Test={len(splits.X_test)}. "
                f"StandardScaler fit strictly on training set."
            )

            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="PASS",
                duration_ms=round(duration, 2),
                summary=summary,
                details={
                    "num_features": len(feature_cols),
                    "aligned_rows": len(df_aligned),
                    "train_size": len(splits.X_train),
                    "val_size": len(splits.X_val),
                    "test_size": len(splits.X_test),
                    "train_date_range": [splits.train_dates[0], splits.train_dates[-1]],
                    "test_date_range": [splits.test_dates[0], splits.test_dates[-1]],
                }
            ))
        except Exception as e:
            duration = (time.time() - start_t) * 1000
            state.error_message = f"Feature Engineering Failed: {e!s}"
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="FAIL",
                duration_ms=round(duration, 2),
                summary=f"Error constructing features: {e!s}",
            ))

        return state
