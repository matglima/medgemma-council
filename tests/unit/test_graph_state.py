"""
Tests for CouncilState schema definition.

TDD: These tests define the expected shape of the shared state object
that flows through the LangGraph state machine. Written BEFORE graph.py.
"""

from typing import get_type_hints


def test_council_state_has_required_keys():
    """CouncilState must contain all required keys from the spec."""
    from graph import CouncilState

    hints = get_type_hints(CouncilState)
    required_keys = {
        "messages",
        "patient_context",
        "medical_images",
        "agent_outputs",
        "debate_history",
        "consensus_reached",
    }
    assert required_keys.issubset(
        hints.keys()
    ), f"Missing keys: {required_keys - hints.keys()}"


def test_council_state_types():
    """CouncilState fields must have the correct types per spec."""
    from graph import CouncilState

    hints = get_type_hints(CouncilState)

    # messages should be a List
    assert "List" in str(hints["messages"]) or "list" in str(hints["messages"])
    # patient_context should be a Dict
    assert "Dict" in str(hints["patient_context"]) or "dict" in str(
        hints["patient_context"]
    )
    # medical_images should be a List
    assert "List" in str(hints["medical_images"]) or "list" in str(
        hints["medical_images"]
    )
    # agent_outputs should be a Dict
    assert "Dict" in str(hints["agent_outputs"]) or "dict" in str(
        hints["agent_outputs"]
    )
    # debate_history should be a List
    assert "List" in str(hints["debate_history"]) or "list" in str(
        hints["debate_history"]
    )
    # consensus_reached should be bool
    assert hints["consensus_reached"] is bool


def test_council_state_can_be_instantiated():
    """CouncilState must be usable as a typed dict (instantiation check)."""
    from graph import CouncilState

    state: CouncilState = {
        "messages": [],
        "patient_context": {"age": 45, "chief_complaint": "headache"},
        "medical_images": [],
        "agent_outputs": {},
        "debate_history": [],
        "consensus_reached": False,
    }
    assert state["consensus_reached"] is False
    assert isinstance(state["agent_outputs"], dict)


def test_council_state_extended_fields():
    """CouncilState should include orchestration fields for debate control."""
    from graph import CouncilState

    hints = get_type_hints(CouncilState)
    # These fields support the debate loop and research retrieval
    assert "research_findings" in hints
    assert "conflict_detected" in hints
    assert "iteration_count" in hints
    assert "final_plan" in hints
