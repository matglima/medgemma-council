"""
Tests for safety override in the LangGraph state machine.

Phase 12: Add a safety_check node that scans agent outputs mid-pipeline
and short-circuits to emergency referral if red flags are found.

TDD: Written BEFORE implementation.
Per CLAUDE.md: "Every agent output must be scanned for Red Flags...
If found, immediately override with an emergency referral message."

Graph topology change:
  specialist -> safety_check
    -> [red_flag] -> emergency_synthesis -> END
    -> [safe] -> conflict_check -> ... (normal flow)
"""

import pytest
from unittest.mock import MagicMock, patch

from graph import (
    CouncilState,
    build_council_graph,
)


# ---------------------------------------------------------------------------
# Node function tests
# ---------------------------------------------------------------------------


class TestSafetyCheckNode:
    """Tests for the safety_check graph node."""

    def test_safety_check_node_exists(self):
        """safety_check_node should be importable from graph."""
        from graph import safety_check_node

    def test_safety_check_no_flags_sets_false(self):
        """When no red flags, safety_check should set red_flag_detected=False."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "CardiologyAgent": "Patient has stable angina. Recommend stress test.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is False
        assert result["emergency_override"] == ""

    def test_safety_check_detects_suicide_risk(self):
        """When specialist output mentions suicide, should flag it."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "PsychiatryAgent": "Patient expresses suicidal ideation with a plan.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is True
        assert "EMERGENCY" in result["emergency_override"]

    def test_safety_check_detects_septic_shock(self):
        """When specialist output mentions septic shock, should flag it."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "EmergencyMedicineAgent": "Patient in septic shock. Lactate > 4.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is True
        assert "EMERGENCY" in result["emergency_override"]

    def test_safety_check_detects_cardiac_arrest(self):
        """When specialist output mentions cardiac arrest, should flag it."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "CardiologyAgent": "Patient found in cardiac arrest. ACLS initiated.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is True

    def test_safety_check_scans_all_agent_outputs(self):
        """Safety check should scan ALL agent outputs, not just one."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "CardiologyAgent": "Stable angina. No acute concerns.",
                "PsychiatryAgent": "Patient is stable, no suicidal ideation.",
                "EmergencyMedicineAgent": "Patient developed anaphylaxis to contrast dye.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is True
        assert "Anaphylaxis" in result["emergency_override"]

    def test_safety_check_includes_flag_labels(self):
        """The emergency override message should include the specific flag labels."""
        from graph import safety_check_node

        state = {
            "agent_outputs": {
                "NeurologyAgent": "Acute ischemic stroke detected. tPA window closing.",
            },
            "red_flag_detected": False,
            "emergency_override": "",
        }
        result = safety_check_node(state)
        assert result["red_flag_detected"] is True
        assert "Acute Stroke" in result["emergency_override"]


class TestEmergencySynthesisNode:
    """Tests for the emergency_synthesis node that short-circuits the graph."""

    def test_emergency_synthesis_node_exists(self):
        """emergency_synthesis_node should be importable from graph."""
        from graph import emergency_synthesis_node

    def test_emergency_synthesis_produces_plan(self):
        """Emergency synthesis should produce a final_plan with the override message."""
        from graph import emergency_synthesis_node

        state = {
            "agent_outputs": {
                "EmergencyMedicineAgent": "Patient in cardiac arrest.",
            },
            "red_flag_detected": True,
            "emergency_override": (
                "EMERGENCY OVERRIDE: Cardiac Arrest. "
                "This case requires IMMEDIATE emergency medical attention."
            ),
            "patient_context": {"chief_complaint": "chest pain"},
            "consensus_reached": False,
            "final_plan": "",
        }
        result = emergency_synthesis_node(state)
        assert "final_plan" in result
        assert "EMERGENCY" in result["final_plan"]
        assert result["consensus_reached"] is True

    def test_emergency_synthesis_includes_specialist_findings(self):
        """Emergency synthesis should include the specialist findings that triggered the flag."""
        from graph import emergency_synthesis_node

        state = {
            "agent_outputs": {
                "PsychiatryAgent": "Patient expresses suicidal ideation.",
            },
            "red_flag_detected": True,
            "emergency_override": "EMERGENCY OVERRIDE: Suicide/Self-Harm Risk.",
            "patient_context": {},
            "consensus_reached": False,
            "final_plan": "",
        }
        result = emergency_synthesis_node(state)
        assert "Suicide" in result["final_plan"] or "suicidal" in result["final_plan"].lower()


# ---------------------------------------------------------------------------
# Conditional edge tests
# ---------------------------------------------------------------------------


class TestSafetyConditionalEdge:
    """Tests for the _should_continue_after_safety conditional edge."""

    def test_edge_function_exists(self):
        """_should_continue_after_safety should be importable from graph."""
        from graph import _should_continue_after_safety

    def test_red_flag_routes_to_emergency(self):
        """When red_flag_detected=True, should route to emergency_synthesis."""
        from graph import _should_continue_after_safety

        state = {"red_flag_detected": True, "emergency_override": "EMERGENCY..."}
        assert _should_continue_after_safety(state) == "emergency_synthesis"

    def test_no_flag_routes_to_conflict_check(self):
        """When red_flag_detected=False, should route to conflict_check."""
        from graph import _should_continue_after_safety

        state = {"red_flag_detected": False, "emergency_override": ""}
        assert _should_continue_after_safety(state) == "conflict_check"


# ---------------------------------------------------------------------------
# Graph topology tests
# ---------------------------------------------------------------------------


class TestGraphWithSafety:
    """Test that the graph includes the safety nodes and edges."""

    def test_graph_has_safety_check_node(self):
        """The compiled graph should contain a safety_check node."""
        graph = build_council_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "safety_check" in node_names

    def test_graph_has_emergency_synthesis_node(self):
        """The compiled graph should contain an emergency_synthesis node."""
        graph = build_council_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "emergency_synthesis" in node_names


# ---------------------------------------------------------------------------
# Full graph execution with safety override
# ---------------------------------------------------------------------------


class TestGraphSafetyExecution:
    """End-to-end tests for the safety override flow."""

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_red_flag_short_circuits_to_emergency(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate
    ):
        """When a specialist detects a red flag, graph should skip debate and go to emergency."""
        sup = MockSupervisor.return_value
        sup.route.return_value = ["EmergencyMedicineAgent"]
        sup.name = "SupervisorAgent"
        # These should NOT be called because safety override skips conflict/debate
        sup.detect_conflict.return_value = False
        sup.synthesize.return_value = {
            "final_plan": "This should NOT appear.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "EmergencyMedicineAgent": "Patient in cardiac arrest. ACLS protocol initiated."
        }

        state = {
            "messages": [{"role": "user", "content": "Evaluate this patient."}],
            "patient_context": {"chief_complaint": "unresponsive", "age": "65", "sex": "M"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
            "red_flag_detected": False,
            "emergency_override": "",
        }

        graph = build_council_graph()
        result = graph.invoke(state)

        assert result["red_flag_detected"] is True
        assert "EMERGENCY" in result["final_plan"]
        assert result["consensus_reached"] is True
        # Debate should never have been called
        mock_debate.assert_not_called()

    @patch("graph._run_debate_round")
    @patch("graph._run_specialists")
    @patch("graph.ResearchAgent")
    @patch("graph.SupervisorAgent")
    def test_safe_case_follows_normal_flow(
        self, MockSupervisor, MockResearch, mock_specialists, mock_debate
    ):
        """When no red flags, graph should proceed through normal conflict/synthesis flow."""
        sup = MockSupervisor.return_value
        sup.route.return_value = ["CardiologyAgent"]
        sup.name = "SupervisorAgent"
        sup.detect_conflict.return_value = False
        sup.synthesize.return_value = {
            "final_plan": "Normal plan. No emergency.",
            "consensus_reached": True,
        }

        mock_specialists.return_value = {
            "CardiologyAgent": "Stable angina. Recommend stress test."
        }

        state = {
            "messages": [{"role": "user", "content": "Evaluate this patient."}],
            "patient_context": {"chief_complaint": "chest pain", "age": "55", "sex": "M"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
            "red_flag_detected": False,
            "emergency_override": "",
        }

        graph = build_council_graph()
        result = graph.invoke(state)

        assert result["red_flag_detected"] is False
        assert "Normal plan" in result["final_plan"]
        assert "EMERGENCY" not in result["final_plan"]
