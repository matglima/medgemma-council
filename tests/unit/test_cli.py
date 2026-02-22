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

    def test_format_text_handles_non_string_agent_outputs(self):
        """Formatter should not crash when an agent output is a list/dict."""
        from council_cli import format_result

        result = {
            "final_plan": "Proceed with CT chest and bronchoscopy.",
            "agent_outputs": {
                "RadiologyAgent": [
                    {"role": "assistant", "content": "No pleural effusion."}
                ]
            },
            "debate_history": [],
            "consensus_reached": True,
            "conflict_detected": False,
            "iteration_count": 0,
            "research_findings": "",
        }

        output = format_result(result, "text")
        assert "RadiologyAgent" in output
        assert "No pleural effusion" in output


class TestVerboseFlag:
    """Tests for verbose logging control in run_council_cli and main()."""

    @patch("council_cli.build_council_graph")
    def test_verbose_true_sets_debug_logging(self, mock_build):
        """verbose=True should configure logging to DEBUG level."""
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [], "patient_context": {}, "medical_images": [],
            "agent_outputs": {}, "debate_history": [],
            "consensus_reached": True, "research_findings": "",
            "conflict_detected": False, "iteration_count": 0,
            "final_plan": "Plan.",
        }
        mock_build.return_value = mock_graph

        with patch("council_cli.logging") as mock_logging:
            run_council_cli(
                age=65, sex="Male", chief_complaint="Chest pain",
                verbose=True,
            )
            mock_logging.basicConfig.assert_called()
            call_kwargs = mock_logging.basicConfig.call_args
            assert call_kwargs[1].get("level") == mock_logging.DEBUG or \
                   (call_kwargs[0] and call_kwargs[0][0] == mock_logging.DEBUG)

    @patch("council_cli.build_council_graph")
    def test_verbose_false_sets_warning_logging(self, mock_build):
        """verbose=False should configure logging to WARNING level."""
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [], "patient_context": {}, "medical_images": [],
            "agent_outputs": {}, "debate_history": [],
            "consensus_reached": True, "research_findings": "",
            "conflict_detected": False, "iteration_count": 0,
            "final_plan": "Plan.",
        }
        mock_build.return_value = mock_graph

        with patch("council_cli.logging") as mock_logging:
            run_council_cli(
                age=65, sex="Male", chief_complaint="Chest pain",
                verbose=False,
            )
            mock_logging.basicConfig.assert_called()
            call_kwargs = mock_logging.basicConfig.call_args
            assert call_kwargs[1].get("level") == mock_logging.WARNING or \
                   (call_kwargs[0] and call_kwargs[0][0] == mock_logging.WARNING)

    @patch("council_cli.build_council_graph")
    def test_verbose_defaults_to_true(self, mock_build):
        """verbose should default to True (high verbosity by default)."""
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [], "patient_context": {}, "medical_images": [],
            "agent_outputs": {}, "debate_history": [],
            "consensus_reached": True, "research_findings": "",
            "conflict_detected": False, "iteration_count": 0,
            "final_plan": "Plan.",
        }
        mock_build.return_value = mock_graph

        # Don't pass verbose — should default to True (DEBUG)
        with patch("council_cli.logging") as mock_logging:
            run_council_cli(
                age=65, sex="Male", chief_complaint="Chest pain",
            )
            mock_logging.basicConfig.assert_called()
            call_kwargs = mock_logging.basicConfig.call_args
            assert call_kwargs[1].get("level") == mock_logging.DEBUG or \
                   (call_kwargs[0] and call_kwargs[0][0] == mock_logging.DEBUG)


class TestTextModelIdFlag:
    """Tests for text_model_id parameter in run_council_cli."""

    @patch("council_cli.build_council_graph")
    def test_text_model_id_sets_env_var(self, mock_build):
        """text_model_id should set MEDGEMMA_TEXT_MODEL_ID env var."""
        import os
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [], "patient_context": {}, "medical_images": [],
            "agent_outputs": {}, "debate_history": [],
            "consensus_reached": True, "research_findings": "",
            "conflict_detected": False, "iteration_count": 0,
            "final_plan": "Plan.",
        }
        mock_build.return_value = mock_graph

        # Capture the env var during graph.invoke
        captured_env = {}
        def capture_invoke(state):
            captured_env["model_id"] = os.environ.get("MEDGEMMA_TEXT_MODEL_ID")
            return mock_graph.invoke.return_value

        mock_graph.invoke.side_effect = capture_invoke

        run_council_cli(
            age=65, sex="Male", chief_complaint="Chest pain",
            text_model_id="google/medgemma-1.5-4b-it",
        )

        assert captured_env["model_id"] == "google/medgemma-1.5-4b-it"

    @patch("council_cli.build_council_graph")
    def test_text_model_id_none_does_not_set_env_var(self, mock_build):
        """When text_model_id is None, MEDGEMMA_TEXT_MODEL_ID should not be set."""
        import os
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [], "patient_context": {}, "medical_images": [],
            "agent_outputs": {}, "debate_history": [],
            "consensus_reached": True, "research_findings": "",
            "conflict_detected": False, "iteration_count": 0,
            "final_plan": "Plan.",
        }
        mock_build.return_value = mock_graph

        # Ensure it's not set before the call
        env_backup = os.environ.pop("MEDGEMMA_TEXT_MODEL_ID", None)
        try:
            captured_env = {}
            def capture_invoke(state):
                captured_env["model_id"] = os.environ.get("MEDGEMMA_TEXT_MODEL_ID")
                return mock_graph.invoke.return_value

            mock_graph.invoke.side_effect = capture_invoke

            run_council_cli(
                age=65, sex="Male", chief_complaint="Chest pain",
            )
            assert captured_env["model_id"] is None
        finally:
            if env_backup is not None:
                os.environ["MEDGEMMA_TEXT_MODEL_ID"] = env_backup

    @patch("council_cli.build_council_graph")
    def test_text_model_id_cleaned_up_after_error(self, mock_build):
        """MEDGEMMA_TEXT_MODEL_ID should be cleaned up even if graph fails."""
        import os
        from council_cli import run_council_cli

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("boom")
        mock_build.return_value = mock_graph

        env_backup = os.environ.pop("MEDGEMMA_TEXT_MODEL_ID", None)
        try:
            run_council_cli(
                age=65, sex="Male", chief_complaint="Chest pain",
                text_model_id="google/medgemma-1.5-4b-it",
            )
            # After the call (which handles the error internally), env should be clean
            assert os.environ.get("MEDGEMMA_TEXT_MODEL_ID") is None
        finally:
            if env_backup is not None:
                os.environ["MEDGEMMA_TEXT_MODEL_ID"] = env_backup


class TestMainArgparse:
    """Tests for argparse CLI changes in main()."""

    @patch("council_cli.run_council_cli")
    @patch("council_cli.format_result")
    def test_model_id_flag_passed_to_run(self, mock_format, mock_run):
        """--model-id flag should be passed as text_model_id to run_council_cli."""
        from council_cli import main

        mock_run.return_value = {"final_plan": "Plan."}
        mock_format.return_value = "output"

        with patch("sys.argv", [
            "council_cli.py",
            "--age", "65", "--sex", "Male", "--complaint", "Chest pain",
            "--model-id", "google/medgemma-1.5-4b-it",
        ]):
            with patch("builtins.print"):
                main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["text_model_id"] == "google/medgemma-1.5-4b-it"

    @patch("council_cli.run_council_cli")
    @patch("council_cli.format_result")
    def test_verbose_defaults_true_in_argparse(self, mock_format, mock_run):
        """--verbose should default to True (no flag needed for debug)."""
        from council_cli import main

        mock_run.return_value = {"final_plan": "Plan."}
        mock_format.return_value = "output"

        with patch("sys.argv", [
            "council_cli.py",
            "--age", "65", "--sex", "Male", "--complaint", "Chest pain",
        ]):
            with patch("builtins.print"):
                main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["verbose"] is True

    @patch("council_cli.run_council_cli")
    @patch("council_cli.format_result")
    def test_quiet_flag_disables_verbose(self, mock_format, mock_run):
        """--quiet flag should set verbose=False."""
        from council_cli import main

        mock_run.return_value = {"final_plan": "Plan."}
        mock_format.return_value = "output"

        with patch("sys.argv", [
            "council_cli.py",
            "--age", "65", "--sex", "Male", "--complaint", "Chest pain",
            "--quiet",
        ]):
            with patch("builtins.print"):
                main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["verbose"] is False
