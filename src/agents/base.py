"""
BaseAgent: Abstract base class for all MedGemma-Council agents.

All specialist agents (Cardiology, Oncology, Pediatrics), the Radiology agent,
the Research agent, and the Supervisor must inherit from this class and
implement the `name` property and `analyze()` method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Abstract base class defining the contract for council agents.

    Args:
        llm: The language model instance (llama-cpp or transformers pipeline).
        system_prompt: Optional persona/instruction prompt for the agent.
    """

    def __init__(self, llm: Any, system_prompt: str = "") -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's display name (e.g., 'CardiologyAgent')."""
        ...

    @abstractmethod
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current council state and return a state update dict.

        Args:
            state: The current CouncilState dictionary.

        Returns:
            A dict with keys to merge back into CouncilState
            (typically updating 'agent_outputs').
        """
        ...
