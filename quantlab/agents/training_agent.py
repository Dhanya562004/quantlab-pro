"""
Model Training Agent for QuantLab Pro.
Trains baseline or PyTorch neural network model with fixed random seed and training metadata.
"""

import time

from quantlab.agents.base import BaseAgent
from quantlab.models.trainer import train_model
from quantlab.orchestration.state import AgentStepTrace, WorkflowState


class ModelTrainingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Model Training Agent",
            agent_role="Trains selected ML model using training split and fixed random seed."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()

        if state.splits is None:
            duration = (time.time() - start_t) * 1000
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="SKIPPED",
                duration_ms=round(duration, 2),
                summary="Skipped due to missing feature splits.",
            ))
            return state

        try:
            artifact, model_obj = train_model(
                model_type=state.selected_model_type,
                splits=state.splits,
                seed=state.user_seed,
            )

            state.trained_artifact = artifact
            state.trained_model_object = model_obj

            duration = (time.time() - start_t) * 1000

            summary = (
                f"Successfully trained '{state.selected_model_type}' model in {artifact.training_duration_sec}s. "
                f"Train Accuracy={artifact.train_accuracy * 100:.1f}%, Val Accuracy={artifact.val_accuracy * 100:.1f}%."
            )

            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="PASS",
                duration_ms=round(duration, 2),
                summary=summary,
                details={
                    "model_type": artifact.model_type,
                    "train_accuracy": artifact.train_accuracy,
                    "val_accuracy": artifact.val_accuracy,
                    "duration_sec": artifact.training_duration_sec,
                    "hyperparameters": artifact.hyperparameters,
                    "seed": artifact.seed,
                }
            ))
        except Exception as e:
            duration = (time.time() - start_t) * 1000
            state.error_message = f"Model Training Failed: {e!s}"
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="FAIL",
                duration_ms=round(duration, 2),
                summary=f"Model training error: {e!s}",
            ))

        return state


TrainingAgent = ModelTrainingAgent
