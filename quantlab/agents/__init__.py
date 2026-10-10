"""Typed Quantitative Research Agents."""

from quantlab.agents.auditor_agent import ExperimentAuditorAgent
from quantlab.agents.base import BaseAgent
from quantlab.agents.evaluation_agent import EvaluationRiskAgent
from quantlab.agents.feature_agent import FeatureEngineeringAgent
from quantlab.agents.planner_agent import QuantResearchPlannerAgent
from quantlab.agents.quality_agent import DataQualityAgent
from quantlab.agents.training_agent import ModelTrainingAgent

# Class Aliases for exact naming alignment
PlannerAgent = QuantResearchPlannerAgent
TrainingAgent = ModelTrainingAgent
EvaluationAgent = EvaluationRiskAgent
AuditorAgent = ExperimentAuditorAgent

__all__ = [
    "BaseAgent",
    "DataQualityAgent",
    "QuantResearchPlannerAgent",
    "PlannerAgent",
    "FeatureEngineeringAgent",
    "ModelTrainingAgent",
    "TrainingAgent",
    "EvaluationRiskAgent",
    "EvaluationAgent",
    "ExperimentAuditorAgent",
    "AuditorAgent",
]

