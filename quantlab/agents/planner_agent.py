"""
Quant Research Planner Agent for QuantLab Pro.
Formulates experiment design, selects model architecture from allowlist, and articulates research rationale.
"""

import time

from quantlab.agents.base import BaseAgent
from quantlab.features.builder import FeatureConfig
from quantlab.orchestration.state import AgentStepTrace, WorkflowState


class QuantResearchPlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="Quant Research Planner Agent",
            agent_role="Selects experiment parameters, verifies target horizon, and articulates research rationale."
        )

    def run(self, state: WorkflowState) -> WorkflowState:
        start_t = time.time()
        
        # Check if validation passed
        if state.validation_report and not state.validation_report.is_valid:
            duration = (time.time() - start_t) * 1000
            state.execution_trace.append(AgentStepTrace(
                step_index=len(state.execution_trace) + 1,
                agent_name=self.agent_name,
                role=self.agent_role,
                status="SKIPPED",
                duration_ms=round(duration, 2),
                summary="Skipped due to upstream data validation failure.",
            ))
            return state

        # Formulate feature & experiment configuration
        target_horizon = state.user_target_horizon
        model_choice = state.user_model_choice.lower().strip()
        
        allowed_models = ["majority_class", "logistic_regression", "pytorch_mlp"]
        if model_choice not in allowed_models:
            model_choice = "logistic_regression"
            
        config = FeatureConfig(
            lags=[1, 2, 3, 5],
            rolling_windows=[5, 10, 20],
            use_rsi=True,
            use_macd=True,
            target_horizon=target_horizon,
            binary_target=True,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        
        rationale = (
            f"Formulated research plan for {state.raw_metadata.get('symbol', 'ASSET')} with target horizon h={target_horizon}. "
            f"Constructing 14 technical features (lags, rolling vol, RSI, MACD). "
            f"Using chronological 60/20/20 train/val/test splits. "
            f"Selected model architecture: '{model_choice}' with seed={state.user_seed}."
        )
        
        state.feature_config = config
        state.selected_model_type = model_choice
        state.research_rationale = rationale
        
        duration = (time.time() - start_t) * 1000
        
        state.execution_trace.append(AgentStepTrace(
            step_index=len(state.execution_trace) + 1,
            agent_name=self.agent_name,
            role=self.agent_role,
            status="PASS",
            duration_ms=round(duration, 2),
            summary=f"Formulated research plan with model '{model_choice}' and target horizon {target_horizon}.",
            details={
                "model_choice": model_choice,
                "target_horizon": target_horizon,
                "feature_lags": config.lags,
                "rolling_windows": config.rolling_windows,
                "split_ratios": [config.train_ratio, config.val_ratio, config.test_ratio],
                "rationale": rationale,
            }
        ))
        
        return state
