"""
Tests for the Streamlit UI backing logic (app.py).

We cannot unit-test Streamlit rendering, so we test:
- State initialization helpers
- Case submission / graph invocation wiring
- File upload processing
- Session state management
- Safety guardrail integration (output scanning)

All graph execution is mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------


class TestInitializeSessionState:
    """Test session state initialization helpers."""

    def test_create_initial_state_returns_valid_council_state(self):
        from app import create_initial_state

        state = create_initial_state()
        assert "messages" in state
        assert "patient_context" in state
        assert "medical_images" in state
        assert "agent_outputs" in state
        assert "debate_history" in state
        assert state["consensus_reached"] is False
        assert state["iteration_count"] == 0
        assert state["final_plan"] == ""

    def test_create_initial_state_messages_empty(self):
        from app import create_initial_state

        state = create_initial_state()
        assert isinstance(state["messages"], list)
        assert len(state["messages"]) == 0


# ---------------------------------------------------------------------------
# Patient context building
# ---------------------------------------------------------------------------


class TestBuildPatientContext:
    """Test patient context construction from form inputs."""

    def test_build_context_with_all_fields(self):
        from app import build_patient_context

        ctx = build_patient_context(
            age=65,
            sex="Male",
            chief_complaint="Chest pain",
            history="Hypertension, diabetes",
            medications="Aspirin, Lisinopril",
        )
        assert ctx["age"] == 65
        assert ctx["sex"] == "Male"
        assert ctx["chief_complaint"] == "Chest pain"
        assert "Hypertension" in ctx["history"]
        assert isinstance(ctx["medications"], list)
        assert "Aspirin" in ctx["medications"]

    def test_build_context_with_minimal_fields(self):
        from app import build_patient_context

        ctx = build_patient_context(
            age=30,
            sex="Female",
            chief_complaint="Headache",
        )
        assert ctx["age"] == 30
        assert ctx["chief_complaint"] == "Headache"
        assert ctx.get("history", "") == ""

    def test_medications_parsed_from_comma_string(self):
        from app import build_patient_context

        ctx = build_patient_context(
            age=50,
            sex="Male",
            chief_complaint="Cough",
            medications="Drug A, Drug B, Drug C",
        )
        assert len(ctx["medications"]) == 3


# ---------------------------------------------------------------------------
# Image upload processing
# ---------------------------------------------------------------------------


class TestProcessUploadedFiles:
    """Test file upload processing for images."""

    def test_process_images_returns_file_paths(self, tmp_path):
        from app import process_uploaded_images

        # Simulate streamlit UploadedFile objects
        mock_file = MagicMock()
        mock_file.name = "xray.png"
        mock_file.read.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        paths = process_uploaded_images([mock_file], upload_dir=str(tmp_path))
        assert len(paths) == 1
        assert "xray.png" in paths[0]

    def test_process_no_images_returns_empty(self, tmp_path):
        from app import process_uploaded_images

        paths = process_uploaded_images([], upload_dir=str(tmp_path))
        assert paths == []

    def test_process_multiple_images(self, tmp_path):
        from app import process_uploaded_images

        files = []
        for name in ["ct_1.png", "ct_2.png", "ct_3.png"]:
            f = MagicMock()
            f.name = name
            f.read.return_value = b"\x89PNG" + b"\x00" * 50
            files.append(f)

        paths = process_uploaded_images(files, upload_dir=str(tmp_path))
        assert len(paths) == 3


# ---------------------------------------------------------------------------
# Council invocation
# ---------------------------------------------------------------------------


class TestRunCouncil:
    """Test the council graph invocation wrapper."""

    @patch("app.build_council_graph")
    def test_run_council_invokes_graph(self, mock_build):
        from app import run_council

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
            "iteration_count": 1,
            "final_plan": "Admit to CCU.",
        }
        mock_build.return_value = mock_graph

        state = {
            "messages": [{"role": "user", "content": "Evaluate patient."}],
            "patient_context": {"age": 65, "chief_complaint": "Chest pain"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        result = run_council(state)
        mock_graph.invoke.assert_called_once_with(state)
        assert result["consensus_reached"] is True
        assert result["final_plan"] == "Admit to CCU."

    @patch("app.build_council_graph")
    def test_run_council_handles_error_gracefully(self, mock_build):
        from app import run_council

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = Exception("Model load failed")
        mock_build.return_value = mock_graph

        state = {
            "messages": [],
            "patient_context": {},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        result = run_council(state)
        assert "error" in result or result.get("final_plan", "").startswith("Error")


# ---------------------------------------------------------------------------
# Safety guardrail on output display
# ---------------------------------------------------------------------------


class TestOutputSafetyIntegration:
    """Test that displayed outputs are scanned for red flags and PII."""

    def test_format_output_scans_red_flags(self):
        from app import format_council_output

        output = format_council_output(
            final_plan="Patient shows signs of cardiac arrest. Begin CPR.",
            agent_outputs={"CardiologyAgent": "VFib detected."},
        )
        # Should contain emergency referral when red flags detected
        assert "EMERGENCY" in output or "911" in output or "emergency" in output.lower()

    def test_format_output_redacts_pii(self):
        from app import format_council_output

        output = format_council_output(
            final_plan="Patient John, SSN 123-45-6789, needs cath lab.",
            agent_outputs={},
        )
        assert "123-45-6789" not in output

    def test_format_output_includes_disclaimer(self):
        from app import format_council_output

        output = format_council_output(
            final_plan="Recommend follow-up in 2 weeks.",
            agent_outputs={"CardiologyAgent": "Stable."},
        )
        assert "disclaimer" in output.lower() or "not a substitute" in output.lower()
