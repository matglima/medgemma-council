"""
Tests for the Gradio UI backing logic (app_gradio.py).

Tests the programmatic interface functions used by Gradio without
starting the actual Gradio server. All graph execution is mocked.

Mirrors the Streamlit test approach: test pure Python helpers, not
the UI framework itself.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------


class TestCreateInitialState:
    """Test Gradio-side state initialization."""

    def test_create_initial_state_returns_valid_dict(self):
        from app_gradio import create_initial_state

        state = create_initial_state()
        assert isinstance(state, dict)
        assert "messages" in state
        assert "patient_context" in state
        assert "agent_outputs" in state
        assert state["consensus_reached"] is False
        assert state["iteration_count"] == 0
        assert state["final_plan"] == ""

    def test_create_initial_state_has_empty_collections(self):
        from app_gradio import create_initial_state

        state = create_initial_state()
        assert state["messages"] == []
        assert state["medical_images"] == []
        assert state["debate_history"] == []
        assert state["agent_outputs"] == {}


# ---------------------------------------------------------------------------
# Patient context building
# ---------------------------------------------------------------------------


class TestBuildPatientContext:
    """Test patient context construction from Gradio form inputs."""

    def test_build_context_with_all_fields(self):
        from app_gradio import build_patient_context

        ctx = build_patient_context(
            age=65,
            sex="Male",
            chief_complaint="Chest pain",
            history="Hypertension",
            medications="Aspirin, Lisinopril",
        )
        assert ctx["age"] == 65
        assert ctx["sex"] == "Male"
        assert ctx["chief_complaint"] == "Chest pain"
        assert "Hypertension" in ctx["history"]
        assert isinstance(ctx["medications"], list)
        assert "Aspirin" in ctx["medications"]
        assert "Lisinopril" in ctx["medications"]

    def test_build_context_with_minimal_fields(self):
        from app_gradio import build_patient_context

        ctx = build_patient_context(
            age=30,
            sex="Female",
            chief_complaint="Headache",
        )
        assert ctx["age"] == 30
        assert ctx["chief_complaint"] == "Headache"
        assert ctx["medications"] == []

    def test_medications_parsed_from_comma_string(self):
        from app_gradio import build_patient_context

        ctx = build_patient_context(
            age=50,
            sex="Other",
            chief_complaint="Cough",
            medications="Drug A, Drug B, Drug C",
        )
        assert len(ctx["medications"]) == 3

    def test_empty_medications_string(self):
        from app_gradio import build_patient_context

        ctx = build_patient_context(
            age=50,
            sex="Male",
            chief_complaint="Cough",
            medications="",
        )
        assert ctx["medications"] == []


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


class TestProcessImages:
    """Test image processing for Gradio file uploads."""

    def test_process_images_returns_paths(self, tmp_path):
        from app_gradio import process_uploaded_images

        # Gradio gives us file paths (strings), not file-like objects
        img_path = tmp_path / "xray.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        paths = process_uploaded_images([str(img_path)])
        assert len(paths) == 1
        assert str(img_path) in paths[0]

    def test_process_no_images_returns_empty(self):
        from app_gradio import process_uploaded_images

        paths = process_uploaded_images(None)
        assert paths == []

    def test_process_empty_list_returns_empty(self):
        from app_gradio import process_uploaded_images

        paths = process_uploaded_images([])
        assert paths == []


# ---------------------------------------------------------------------------
# Council invocation
# ---------------------------------------------------------------------------


class TestRunCouncilGradio:
    """Test the Gradio council execution wrapper."""

    @patch("app_gradio.build_council_graph")
    def test_run_council_returns_formatted_output(self, mock_build):
        from app_gradio import run_council_analysis

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
            "final_plan": "Admit to CCU. Start heparin.",
        }
        mock_build.return_value = mock_graph

        plan, specialists, status = run_council_analysis(
            age=65,
            sex="Male",
            chief_complaint="Chest pain",
            history="Hypertension",
            medications="Aspirin",
            images=None,
        )
        # Final plan should appear in output (with safety applied)
        assert "Admit to CCU" in plan
        # Specialist findings should be returned
        assert "CardiologyAgent" in specialists
        # Status should report consensus
        assert "Yes" in status or "Consensus" in status

    @patch("app_gradio.build_council_graph")
    def test_run_council_handles_error(self, mock_build):
        from app_gradio import run_council_analysis

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("GPU OOM")
        mock_build.return_value = mock_graph

        plan, specialists, status = run_council_analysis(
            age=65,
            sex="Male",
            chief_complaint="Chest pain",
        )
        assert "Error" in plan or "error" in plan.lower()

    @patch("app_gradio.build_council_graph")
    def test_run_council_requires_chief_complaint(self, mock_build):
        from app_gradio import run_council_analysis

        plan, specialists, status = run_council_analysis(
            age=65,
            sex="Male",
            chief_complaint="",
        )
        # Should not invoke graph with empty complaint
        mock_build.assert_not_called()
        assert "complaint" in plan.lower() or "required" in plan.lower()


# ---------------------------------------------------------------------------
# Output formatting with safety
# ---------------------------------------------------------------------------


class TestFormatOutput:
    """Test that Gradio output is safety-checked."""

    def test_format_output_redacts_pii(self):
        from app_gradio import format_plan_output

        output = format_plan_output(
            "Patient SSN 123-45-6789 needs cath lab."
        )
        assert "123-45-6789" not in output

    def test_format_output_triggers_emergency_on_red_flags(self):
        from app_gradio import format_plan_output

        output = format_plan_output(
            "Patient in cardiac arrest. Begin CPR."
        )
        assert "EMERGENCY" in output or "911" in output or "emergency" in output.lower()

    def test_format_output_includes_disclaimer(self):
        from app_gradio import format_plan_output

        output = format_plan_output("Recommend follow-up in 2 weeks.")
        assert "disclaimer" in output.lower() or "not a substitute" in output.lower()

    def test_format_specialists_output(self):
        from app_gradio import format_specialists_output

        output = format_specialists_output({
            "CardiologyAgent": "Elevated troponin.",
            "RadiologyAgent": "Pulmonary infiltrate seen.",
        })
        assert "CardiologyAgent" in output
        assert "RadiologyAgent" in output
        assert "Elevated troponin" in output

    def test_format_specialists_empty(self):
        from app_gradio import format_specialists_output

        output = format_specialists_output({})
        assert output == "" or "no specialist" in output.lower()


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------


class TestBuildGradioApp:
    """Test that the Gradio app object is created correctly."""

    def test_build_app_returns_blocks_instance(self):
        import gradio as gr
        from app_gradio import build_gradio_app

        app = build_gradio_app()
        assert isinstance(app, gr.Blocks)

    def test_build_app_is_callable(self):
        from app_gradio import build_gradio_app

        app = build_gradio_app()
        # Gradio Blocks objects have a launch method
        assert hasattr(app, "launch")
