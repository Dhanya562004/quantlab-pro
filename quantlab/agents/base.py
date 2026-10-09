"""
Base Agent Interface for QuantLab Pro.
"""

from abc import ABC, abstractmethod

from quantlab.orchestration.state import WorkflowState


class BaseAgent(ABC):
    """Abstract base class for typed quantitative research agents."""
    def __init__(self, agent_name: str, agent_role: str):
        self.agent_name = agent_name
        self.agent_role = agent_role

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:
        """Execute agent task, update state, and return updated state."""
