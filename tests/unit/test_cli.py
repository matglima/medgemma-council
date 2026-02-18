"""
Tests for the CLI interface (council_cli.py).

Tests the programmatic API and output formatting without loading
any real models. All graph execution is mocked.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestBuildState:
    """Test state construction from CLI parameters."""

    def test_build_state_minimal(self):
        from council_cli import build_state

        state = build_state(age=65, sex="Male", chief_complaint="Chest pain")
        assert state["patient_context"]["age"] == 65
        assert state["patient_context"]["sex"] == "Male"
        assert state["patient_context"]["chief_complaint"] == "Chest pain"
        assert state["iteration_count"] == 0
        assert state["final_plan"] == ""

    def test_build_state_with_all_fields(self):
        from council_cli import build_state

        state = build_state(
            age=5,
            sex="Female",
            chief_complaint="Fever",
            history="None",
            medications=["Tylenol"],
            vitals={"temp": 102.5},
            labs={"wbc": 15000},
            image_paths=["/tmp/xray.png"],
        )
        assert state["patient_context"]["medications"] == ["Tylenol"]
        assert state["patient_context"]["vitals"]["temp"] == 102.5
        assert state["patient_context"]["labs"]["wbc"] == 15000
        assert state["medical_images"] == ["/tmp/xray.png"]

    def test_build_state_defaults(self):
        from council_cli import build_state

        state = build_state(age=30, sex="Other", chief_complaint="Headache")
        assert state["patient_context"]["medications"] == []
        assert state["medical_images"] == []
        assert state["consensus_reached"] is False


class TestRunCouncilCli:
    """Test the programmatic council invocation."""

    @patch("council_cli.build_council_graph")
    def test_run_returns_result(self, mock_build):
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [],
            "patient_context": {},
            "medical_images": [],
            "agent_outputs": {"CardiologyAgent": "ACS likely."},
            "debate_history": [],
            "consensus_reached": True,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "Admit to CCU.",
        }
        mock_build.return_value = mock_graph

        result = run_council_cli(
            age=65, sex="Male", chief_complaint="Chest pain"
        )
        assert result["final_plan"] == "Admit to CCU."
        assert result["consensus_reached"] is True

    @patch("council_cli.build_council_graph")
    def test_run_handles_error(self, mock_build):
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Model not found")
        mock_build.return_value = mock_graph

        result = run_council_cli(
            age=65, sex="Male", chief_complaint="Chest pain"
        )
        assert "Error" in result["final_plan"]
        assert result["consensus_reached"] is False


class TestFormatResult:
    """Test output formatting with safety guardrails."""

    def test_format_text_includes_plan(self):
        from council_cli import format_result

        result = {
            "final_plan": "Admit patient. Start IV fluids.",
            "agent_outputs": {"CardiologyAgent": "Elevated troponin."},
            "debate_history": [],
            "consensus_reached": True,
            "conflict_detected": False,
            "iteration_count": 0,
            "research_findings": "",
        }
        output = format_result(result, "text")
        assert "Admit patient" in output
        assert "CLINICAL MANAGEMENT PLAN" in output
        assert "DISCLAIMER" in output or "not a substitute" in output.lower()

    def test_format_json_is_valid(self):
        from council_cli import format_result

        result = {
            "final_plan": "Follow up in 2 weeks.",
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": True,
            "conflict_detected": False,
            "iteration_count": 0,
            "research_findings": "",
        }
        output = format_result(result, "json")
        parsed = json.loads(output)
        assert parsed["final_plan"] == "Follow up in 2 weeks."
        assert parsed["consensus_reached"] is True

    def test_format_text_redacts_pii(self):
        from council_cli import format_result

        result = {
            "final_plan": "Patient SSN 123-45-6789 needs cath lab.",
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": True,
            "conflict_detected": False,
            "iteration_count": 0,
            "research_findings": "",
        }
        output = format_result(result, "text")
        assert "123-45-6789" not in output

    def test_format_text_shows_emergency_on_red_flags(self):
        from council_cli import format_result

        result = {
            "final_plan": "Patient in cardiac arrest. Begin CPR.",
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "conflict_detected": False,
            "iteration_count": 0,
            "research_findings": "",
        }
        output = format_result(result, "text")
        assert "EMERGENCY" in output

    def test_format_text_shows_debate_history(self):
        from council_cli import format_result

        result = {
            "final_plan": "Consensus plan.",
            "agent_outputs": {},
            "debate_history": ["Round 1: Agents disagreed.", "Round 2: Consensus reached."],
            "consensus_reached": True,
            "conflict_detected": False,
            "iteration_count": 2,
            "research_findings": "",
        }
        output = format_result(result, "text")
        assert "DEBATE HISTORY" in output
        assert "Round 1" in output
