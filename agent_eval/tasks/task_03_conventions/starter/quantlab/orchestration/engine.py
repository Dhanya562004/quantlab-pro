"""Multi-agent orchestrator engine."""

from quantlab.agents.existing_agents import (
    AuditorAgent,
    DataQualityAgent,
    EvaluationAgent,
    FeatureEngineeringAgent,
    PlannerAgent,
    TrainingAgent,
)
from quantlab.orchestration.state import WorkflowState


class MultiAgentOrchestrator:
    """Coordinates execution across agents in deterministic sequence."""

    def __init__(self):
        self.agents = [
            DataQualityAgent(),
            PlannerAgent(),
            FeatureEngineeringAgent(),
            TrainingAgent(),
            EvaluationAgent(),
            AuditorAgent(),
        ]

    def run(self, initial_state: WorkflowState | None = None) -> WorkflowState:
        state = initial_state or WorkflowState()
        for agent in self.agents:
            state = agent.run(state)
        state.is_completed = True
        return state

