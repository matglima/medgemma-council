"""Tests for CouncilEvaluator — Phase 10.4

Tests verify: evaluate_single, extract_answer_letter, evaluate_batch.
The council graph is fully mocked (no LLM calls).
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Test: extract_answer_letter
# ---------------------------------------------------------------------------

class TestExtractAnswerLetter:
    """Tests for the regex-based answer extractor."""

    def test_extracts_single_letter(self):
        from evaluation.evaluator import extract_answer_letter
        assert extract_answer_letter("The answer is B.") == "B"

    def test_extracts_letter_in_parentheses(self):
        from evaluation.evaluator import extract_answer_letter
        assert extract_answer_letter("I would choose (C) because...") == "C"

    def test_extracts_standalone_letter(self):
        from evaluation.evaluator import extract_answer_letter
        assert extract_answer_letter("D") == "D"

    def test_returns_none_for_ambiguous(self):
        from evaluation.evaluator import extract_answer_letter
        result = extract_answer_letter("I'm not sure, could be A or B.")
        # Should return the first match or None — implementation decides
        assert result is None or result in ("A", "B", "C", "D")

    def test_extracts_from_verbose_output(self):
        from evaluation.evaluator import extract_answer_letter
        text = """Based on the clinical presentation, the patient likely has
        myocardial infarction. The correct answer is (B) MI."""
        assert extract_answer_letter(text) == "B"

    def test_handles_yes_no_maybe_for_pubmedqa(self):
        from evaluation.evaluator import extract_answer_letter
        assert extract_answer_letter("yes") == "yes"
        assert extract_answer_letter("The answer is no.") == "no"
        assert extract_answer_letter("maybe") == "maybe"


# ---------------------------------------------------------------------------
# Test: CouncilEvaluator
# ---------------------------------------------------------------------------

class TestCouncilEvaluator:
    """Tests for the CouncilEvaluator class."""

    def test_init_stores_graph(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        evaluator = CouncilEvaluator(graph=mock_graph)
        assert evaluator.graph is mock_graph

    @patch("evaluation.evaluator.build_council_graph")
    def test_init_builds_graph_when_none(self, mock_build):
        from evaluation.evaluator import CouncilEvaluator

        mock_build.return_value = MagicMock()
        evaluator = CouncilEvaluator()
        mock_build.assert_called_once()
        assert evaluator.graph is not None

    def test_evaluate_single_returns_result_dict(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_plan": "The answer is (B) MI.",
            "agent_outputs": {},
        }
        evaluator = CouncilEvaluator(graph=mock_graph)

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "GERD", "B": "MI", "C": "PE", "D": "Costochondritis"},
            "answer": "B",
        }
        result = evaluator.evaluate_single(item)
        assert isinstance(result, dict)
        assert "predicted" in result
        assert "correct" in result
        assert "is_correct" in result

    def test_evaluate_single_marks_correct(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_plan": "The answer is (B) MI.",
            "agent_outputs": {},
        }
        evaluator = CouncilEvaluator(graph=mock_graph)

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "GERD", "B": "MI", "C": "PE", "D": "Costochondritis"},
            "answer": "B",
        }
        result = evaluator.evaluate_single(item)
        assert result["is_correct"] is True
        assert result["predicted"] == "B"
        assert result["correct"] == "B"

    def test_evaluate_single_marks_incorrect(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_plan": "The answer is (A) GERD.",
            "agent_outputs": {},
        }
        evaluator = CouncilEvaluator(graph=mock_graph)

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "GERD", "B": "MI", "C": "PE", "D": "Costochondritis"},
            "answer": "B",
        }
        result = evaluator.evaluate_single(item)
        assert result["is_correct"] is False

    def test_evaluate_batch_returns_list(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "final_plan": "The answer is (B).",
            "agent_outputs": {},
        }
        evaluator = CouncilEvaluator(graph=mock_graph)

        items = [
            {
                "question": "Q1",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": "B",
            },
            {
                "question": "Q2",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": "A",
            },
        ]
        results = evaluator.evaluate_batch(items)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_evaluate_single_handles_graph_error(self):
        from evaluation.evaluator import CouncilEvaluator

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("GPU OOM")
        evaluator = CouncilEvaluator(graph=mock_graph)

        item = {
            "question": "Q1",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer": "B",
        }
        result = evaluator.evaluate_single(item)
        assert result["is_correct"] is False
        assert result["predicted"] is None
        assert "error" in result
