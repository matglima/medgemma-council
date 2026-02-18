"""
Tests for BaseAgent abstract base class.

TDD: Defines the contract that all agents (Cardio, Onco, Peds, Radiology,
Research, Supervisor) must fulfill. Written BEFORE base.py.
"""

import pytest
from unittest.mock import MagicMock


def test_base_agent_is_abstract():
    """BaseAgent cannot be instantiated directly."""
    from agents.base import BaseAgent

    with pytest.raises(TypeError):
        BaseAgent(llm=MagicMock())


def test_base_agent_requires_analyze_method():
    """Subclasses must implement analyze()."""
    from agents.base import BaseAgent

    class IncompleteAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "Incomplete"

    with pytest.raises(TypeError):
        IncompleteAgent(llm=MagicMock())


def test_base_agent_requires_name_property():
    """Subclasses must implement the name property."""
    from agents.base import BaseAgent

    class IncompleteAgent(BaseAgent):
        def analyze(self, state):
            return {}

    with pytest.raises(TypeError):
        IncompleteAgent(llm=MagicMock())


def test_concrete_agent_can_be_instantiated(mock_llm):
    """A fully implemented subclass should instantiate without error."""
    from agents.base import BaseAgent

    class ConcreteAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "TestAgent"

        def analyze(self, state):
            return {"agent_outputs": {self.name: "test output"}}

    agent = ConcreteAgent(llm=mock_llm)
    assert agent.name == "TestAgent"


def test_concrete_agent_analyze_returns_state_update(mock_llm, sample_patient_context):
    """analyze() must return a dict suitable for merging into CouncilState."""
    from agents.base import BaseAgent

    class ConcreteAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "TestAgent"

        def analyze(self, state):
            return {"agent_outputs": {self.name: "analysis complete"}}

    agent = ConcreteAgent(llm=mock_llm)
    state = {
        "messages": [],
        "patient_context": sample_patient_context,
        "medical_images": [],
        "agent_outputs": {},
        "debate_history": [],
        "consensus_reached": False,
        "research_findings": "",
        "conflict_detected": False,
        "iteration_count": 0,
        "final_plan": "",
    }
    result = agent.analyze(state)
    assert "agent_outputs" in result
    assert "TestAgent" in result["agent_outputs"]


def test_base_agent_stores_llm_reference(mock_llm):
    """BaseAgent.__init__ must store the LLM reference for subclass use."""
    from agents.base import BaseAgent

    class ConcreteAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "TestAgent"

        def analyze(self, state):
            return {}

    agent = ConcreteAgent(llm=mock_llm)
    assert agent.llm is mock_llm


def test_base_agent_has_system_prompt(mock_llm):
    """BaseAgent should accept and store an optional system_prompt."""
    from agents.base import BaseAgent

    class ConcreteAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "TestAgent"

        def analyze(self, state):
            return {}

    prompt = "You are a board-certified cardiologist."
    agent = ConcreteAgent(llm=mock_llm, system_prompt=prompt)
    assert agent.system_prompt == prompt
