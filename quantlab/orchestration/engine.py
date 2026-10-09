"""
Multi-Agent Orchestrator Engine for QuantLab Pro.
Enforces typed state transitions, step sequence execution, and failure handling.
Includes extensible LLMProvider interface for optional natural language reasoning.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quantlab.agents.auditor_agent import ExperimentAuditorAgent
from quantlab.agents.evaluation_agent import EvaluationRiskAgent
from quantlab.agents.feature_agent import FeatureEngineeringAgent
from quantlab.agents.planner_agent import QuantResearchPlannerAgent
from quantlab.agents.quality_agent import DataQualityAgent
from quantlab.agents.training_agent import ModelTrainingAgent
from quantlab.orchestration.state import WorkflowState


class BaseLLMProvider(ABC):
    """Abstract interface for optional LLM reasoning integration."""
    @abstractmethod
    def generate_summary(self, prompt: str) -> str:
        pass


class DeterministicMockLLMProvider(BaseLLMProvider):
    """Default rule-based deterministic summary provider requiring no external API keys."""
    def generate_summary(self, prompt: str) -> str:
        return "[Deterministic Research Summary]: Execution validated. Analysis derived strictly from empirical quantitative features and held-out test evaluation."


class MultiAgentOrchestrator:
    """
    Orchestrates the sequential execution of the 6 specialized quantitative ML research agents.
    """
    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        self.agents = [
            DataQualityAgent(),
            QuantResearchPlannerAgent(),
            FeatureEngineeringAgent(),
            ModelTrainingAgent(),
            EvaluationRiskAgent(),
            ExperimentAuditorAgent(),
        ]
        self.llm_provider = llm_provider or DeterministicMockLLMProvider()

    def run_pipeline(
        self,
        df_raw: pd.DataFrame,
        raw_metadata: dict[str, Any],
        model_choice: str = "logistic_regression",
        seed: int = 42,
        target_horizon: int = 1,
        transaction_cost_bps: float = 10.0,
    ) -> WorkflowState:
        """
        Execute end-to-end multi-agent quantitative workflow.

        Args:
            df_raw: Raw OHLCV DataFrame.
            raw_metadata: Provenance metadata dictionary.
            model_choice: 'majority_class', 'logistic_regression', or 'pytorch_mlp'.
            seed: Fixed random seed.
            target_horizon: Prediction forecast horizon in bars.
            transaction_cost_bps: Transaction cost per trade in bps.

        Returns:
            Final WorkflowState object containing execution trace and results.
        """
        state = WorkflowState(
            df_raw=df_raw,
            raw_metadata=raw_metadata,
            user_model_choice=model_choice,
            user_seed=seed,
            user_target_horizon=target_horizon,
            user_transaction_cost_bps=transaction_cost_bps,
        )

        for agent in self.agents:
            # Execute step
            state = agent.run(state)

            # Check for halting conditions
            if state.error_message:
                state.is_completed = False
                break

        return state
