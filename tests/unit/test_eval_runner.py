"""Tests for the evaluation CLI runner — Phase 10.8

Tests verify: argument parsing, main flow, JSON output writing.
All heavy compute (graph, datasets) is fully mocked.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Test: parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_benchmark_is_medqa(self):
        from evaluation.runner import parse_args

        args = parse_args([])
        assert args.benchmark == "medqa"

    def test_benchmark_flag(self):
        from evaluation.runner import parse_args

        args = parse_args(["--benchmark", "pubmedqa"])
        assert args.benchmark == "pubmedqa"

    def test_limit_flag(self):
        from evaluation.runner import parse_args

        args = parse_args(["--limit", "50"])
        assert args.limit == 50

    def test_output_flag(self):
        from evaluation.runner import parse_args

        args = parse_args(["--output", "results.json"])
        assert args.output == "results.json"

    def test_default_limit_is_none(self):
        from evaluation.runner import parse_args

        args = parse_args([])
        assert args.limit is None

    def test_specialty_flag(self):
        from evaluation.runner import parse_args

        args = parse_args(["--benchmark", "medmcqa", "--specialty", "Pediatrics"])
        assert args.specialty == "Pediatrics"


# ---------------------------------------------------------------------------
# Test: run_evaluation (main flow)
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    """Tests for the main evaluation orchestrator."""

    @patch("evaluation.runner.CouncilEvaluator")
    @patch("evaluation.runner.load_medqa")
    def test_run_evaluation_calls_loader(self, mock_load, mock_evaluator_cls):
        from evaluation.runner import run_evaluation

        mock_load.return_value = [
            {"question": "Q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B"},
        ]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_batch.return_value = [{"is_correct": True, "predicted": "B", "correct": "B"}]
        mock_evaluator_cls.return_value = mock_evaluator

        result = run_evaluation(benchmark="medqa", limit=1)
        mock_load.assert_called_once_with(limit=1)
        assert "report" in result

    @patch("evaluation.runner.CouncilEvaluator")
    @patch("evaluation.runner.load_pubmedqa")
    def test_run_evaluation_pubmedqa(self, mock_load, mock_evaluator_cls):
        from evaluation.runner import run_evaluation

        mock_load.return_value = [
            {"question": "Q1", "context": "ctx", "answer": "yes"},
        ]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_batch.return_value = [{"is_correct": True, "predicted": "yes", "correct": "yes"}]
        mock_evaluator_cls.return_value = mock_evaluator

        result = run_evaluation(benchmark="pubmedqa", limit=1)
        mock_load.assert_called_once_with(limit=1)
        assert result["report"]["benchmark"] == "pubmedqa"

    @patch("evaluation.runner.CouncilEvaluator")
    @patch("evaluation.runner.load_medmcqa")
    def test_run_evaluation_medmcqa(self, mock_load, mock_evaluator_cls):
        from evaluation.runner import run_evaluation

        mock_load.return_value = [
            {"question": "Q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A", "subject": "Pediatrics"},
        ]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_batch.return_value = [{"is_correct": True, "predicted": "A", "correct": "A", "subject": "Pediatrics"}]
        mock_evaluator_cls.return_value = mock_evaluator

        result = run_evaluation(benchmark="medmcqa", limit=1)
        mock_load.assert_called_once_with(limit=1)

    @patch("evaluation.runner.CouncilEvaluator")
    @patch("evaluation.runner.load_medqa")
    def test_run_evaluation_returns_report_and_results(self, mock_load, mock_evaluator_cls):
        from evaluation.runner import run_evaluation

        mock_load.return_value = [
            {"question": "Q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B"},
        ]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_batch.return_value = [{"is_correct": True, "predicted": "B", "correct": "B"}]
        mock_evaluator_cls.return_value = mock_evaluator

        result = run_evaluation(benchmark="medqa")
        assert "report" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    @patch("evaluation.runner.CouncilEvaluator")
    @patch("evaluation.runner.load_medqa")
    def test_run_evaluation_writes_json_output(self, mock_load, mock_evaluator_cls, tmp_path):
        from evaluation.runner import run_evaluation

        mock_load.return_value = [
            {"question": "Q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "B"},
        ]
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_batch.return_value = [{"is_correct": True, "predicted": "B", "correct": "B"}]
        mock_evaluator_cls.return_value = mock_evaluator

        output_file = tmp_path / "results.json"
        run_evaluation(benchmark="medqa", output=str(output_file))

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "report" in data
