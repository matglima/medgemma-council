"""
Tests for Phase 15: PMC-Patients evaluation.

Sub-module 1: PMC-Patients dataset loader
Sub-module 2: Retrieval metrics (MRR, NDCG@10)
Sub-module 3: LLM-as-Judge evaluation

TDD: Written BEFORE implementation.
Per CLAUDE.md: "Write failing tests BEFORE implementation code for every module."
"""

import math
import pytest
from unittest.mock import MagicMock, patch, call


# ===========================================================================
# Sub-module 1: PMC-Patients dataset loader
# ===========================================================================


class TestLoadPmcPatients:
    """Tests for the PMC-Patients dataset loader."""

    def test_import(self):
        """load_pmc_patients should be importable from evaluation.pmc_patients."""
        from evaluation.pmc_patients import load_pmc_patients

    @patch("evaluation.pmc_patients.hf_load_dataset")
    def test_returns_list_of_dicts(self, mock_load):
        """load_pmc_patients should return a list of dicts."""
        from evaluation.pmc_patients import load_pmc_patients

        mock_load.return_value = {
            "train": [
                {
                    "patient_id": "PMC001",
                    "patient": "A 55-year-old male with chest pain...",
                    "title": "Acute MI case",
                    "patient_uid": "uid001",
                    "PMID": 12345678,
                },
            ]
        }

        result = load_pmc_patients(limit=1)
        assert isinstance(result, list)
        assert len(result) == 1

    @patch("evaluation.pmc_patients.hf_load_dataset")
    def test_has_required_keys(self, mock_load):
        """Each item should have patient_id, patient_text, and title."""
        from evaluation.pmc_patients import load_pmc_patients

        mock_load.return_value = {
            "train": [
                {
                    "patient_id": "PMC001",
                    "patient": "A 55-year-old male...",
                    "title": "Acute MI case",
                    "patient_uid": "uid001",
                    "PMID": 12345678,
                },
            ]
        }

        result = load_pmc_patients(limit=1)
        item = result[0]
        assert "patient_id" in item
        assert "patient_text" in item
        assert "title" in item

    @patch("evaluation.pmc_patients.hf_load_dataset")
    def test_respects_limit(self, mock_load):
        """limit parameter should cap the number of returned items."""
        from evaluation.pmc_patients import load_pmc_patients

        mock_load.return_value = {
            "train": [
                {
                    "patient_id": f"PMC{i:03d}",
                    "patient": f"Patient {i} text...",
                    "title": f"Case {i}",
                    "patient_uid": f"uid{i:03d}",
                    "PMID": 12345678 + i,
                }
                for i in range(100)
            ]
        }

        result = load_pmc_patients(limit=5)
        assert len(result) == 5

    @patch("evaluation.pmc_patients.hf_load_dataset")
    def test_no_limit_returns_all(self, mock_load):
        """Without limit, should return all items."""
        from evaluation.pmc_patients import load_pmc_patients

        mock_load.return_value = {
            "train": [
                {
                    "patient_id": f"PMC{i:03d}",
                    "patient": f"Patient {i}",
                    "title": f"Case {i}",
                    "patient_uid": f"uid{i:03d}",
                    "PMID": 12345678 + i,
                }
                for i in range(10)
            ]
        }

        result = load_pmc_patients()
        assert len(result) == 10


class TestFormatPmcPatientPrompt:
    """Tests for formatting PMC-Patients items into prompts."""

    def test_import(self):
        """format_pmc_patient_prompt should be importable."""
        from evaluation.pmc_patients import format_pmc_patient_prompt

    def test_prompt_contains_patient_text(self):
        """Formatted prompt should include the patient description."""
        from evaluation.pmc_patients import format_pmc_patient_prompt

        item = {
            "patient_id": "PMC001",
            "patient_text": "A 55-year-old male presenting with chest pain.",
            "title": "Acute MI",
        }
        prompt = format_pmc_patient_prompt(item)
        assert "55-year-old male" in prompt
        assert "chest pain" in prompt


# ===========================================================================
# Sub-module 2: Retrieval metrics (MRR, NDCG@10)
# ===========================================================================


class TestMRR:
    """Tests for Mean Reciprocal Rank computation."""

    def test_import(self):
        """compute_mrr should be importable from evaluation.retrieval_metrics."""
        from evaluation.retrieval_metrics import compute_mrr

    def test_perfect_retrieval(self):
        """When the relevant item is always rank 1, MRR should be 1.0."""
        from evaluation.retrieval_metrics import compute_mrr

        # Each query result: list of (item_id, is_relevant) pairs
        results = [
            [("doc1", True), ("doc2", False), ("doc3", False)],
            [("docA", True), ("docB", False)],
        ]
        assert compute_mrr(results) == 1.0

    def test_second_rank(self):
        """When relevant item is always rank 2, MRR should be 0.5."""
        from evaluation.retrieval_metrics import compute_mrr

        results = [
            [("doc1", False), ("doc2", True), ("doc3", False)],
            [("docA", False), ("docB", True)],
        ]
        assert compute_mrr(results) == 0.5

    def test_mixed_ranks(self):
        """MRR should be the mean of 1/rank for each query."""
        from evaluation.retrieval_metrics import compute_mrr

        results = [
            [("doc1", True), ("doc2", False)],   # rank 1 -> 1/1
            [("doc1", False), ("doc2", True)],    # rank 2 -> 1/2
        ]
        assert compute_mrr(results) == pytest.approx(0.75)

    def test_no_relevant_results(self):
        """When no relevant items found, MRR should be 0.0."""
        from evaluation.retrieval_metrics import compute_mrr

        results = [
            [("doc1", False), ("doc2", False)],
        ]
        assert compute_mrr(results) == 0.0

    def test_empty_results(self):
        """Empty results should return 0.0."""
        from evaluation.retrieval_metrics import compute_mrr

        assert compute_mrr([]) == 0.0


class TestNDCG:
    """Tests for Normalized Discounted Cumulative Gain at k."""

    def test_import(self):
        """compute_ndcg should be importable from evaluation.retrieval_metrics."""
        from evaluation.retrieval_metrics import compute_ndcg

    def test_perfect_ranking(self):
        """Perfect ranking should have NDCG = 1.0."""
        from evaluation.retrieval_metrics import compute_ndcg

        # relevance_scores: list of (predicted_ranking_scores, true_relevance_scores)
        # When top items are the most relevant, NDCG should be 1.0
        results = [
            {"retrieved_relevance": [3, 2, 1, 0], "ideal_relevance": [3, 2, 1, 0]},
        ]
        assert compute_ndcg(results, k=4) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Reversed ranking should have NDCG < 1.0."""
        from evaluation.retrieval_metrics import compute_ndcg

        results = [
            {"retrieved_relevance": [0, 1, 2, 3], "ideal_relevance": [3, 2, 1, 0]},
        ]
        ndcg = compute_ndcg(results, k=4)
        assert ndcg < 1.0
        assert ndcg > 0.0

    def test_default_k_is_10(self):
        """Default k should be 10."""
        from evaluation.retrieval_metrics import compute_ndcg

        results = [
            {"retrieved_relevance": [1] * 10, "ideal_relevance": [1] * 10},
        ]
        # Should not raise with default k
        ndcg = compute_ndcg(results)
        assert ndcg == pytest.approx(1.0)

    def test_empty_results(self):
        """Empty results should return 0.0."""
        from evaluation.retrieval_metrics import compute_ndcg

        assert compute_ndcg([]) == 0.0


# ===========================================================================
# Sub-module 3: LLM-as-Judge evaluation
# ===========================================================================


class TestLLMJudge:
    """Tests for the LLM-as-a-Judge evaluator."""

    def test_import(self):
        """LLMJudge should be importable from evaluation.llm_judge."""
        from evaluation.llm_judge import LLMJudge

    def test_init_stores_llm(self):
        """LLMJudge should store the LLM callable."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock()
        judge = LLMJudge(llm=mock_llm)
        assert judge.llm is mock_llm

    def test_evaluate_plan_returns_score_dict(self):
        """evaluate_plan() should return a dict with score and rationale."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock(return_value={
            "choices": [{"text": "Score: 4/5\nRationale: Good plan with minor gaps."}]
        })
        judge = LLMJudge(llm=mock_llm)

        result = judge.evaluate_plan(
            patient_context={"chief_complaint": "chest pain", "age": "55"},
            clinical_plan="Admit to CCU. Start heparin drip. Cardiology consult.",
        )

        assert isinstance(result, dict)
        assert "score" in result
        assert "rationale" in result

    def test_evaluate_plan_score_is_numeric(self):
        """The score should be a numeric value."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock(return_value={
            "choices": [{"text": "Score: 4/5\nRationale: Solid plan."}]
        })
        judge = LLMJudge(llm=mock_llm)

        result = judge.evaluate_plan(
            patient_context={"chief_complaint": "chest pain"},
            clinical_plan="Admit. Start treatment.",
        )

        assert isinstance(result["score"], (int, float))

    def test_evaluate_plan_calls_llm(self):
        """evaluate_plan() should call the LLM with a judging prompt."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock(return_value={
            "choices": [{"text": "Score: 3/5\nRationale: Adequate."}]
        })
        judge = LLMJudge(llm=mock_llm)

        judge.evaluate_plan(
            patient_context={"chief_complaint": "headache"},
            clinical_plan="Prescribe acetaminophen.",
        )

        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "headache" in prompt.lower() or "plan" in prompt.lower()

    def test_evaluate_batch_returns_list(self):
        """evaluate_batch() should return a list of score dicts."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock(return_value={
            "choices": [{"text": "Score: 4/5\nRationale: Good."}]
        })
        judge = LLMJudge(llm=mock_llm)

        cases = [
            {
                "patient_context": {"chief_complaint": "chest pain"},
                "clinical_plan": "Admit to CCU.",
            },
            {
                "patient_context": {"chief_complaint": "headache"},
                "clinical_plan": "Prescribe acetaminophen.",
            },
        ]

        results = judge.evaluate_batch(cases)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_evaluate_plan_handles_llm_error(self):
        """If the LLM fails, should return a dict with score=0 and error."""
        from evaluation.llm_judge import LLMJudge

        mock_llm = MagicMock(side_effect=RuntimeError("Model OOM"))
        judge = LLMJudge(llm=mock_llm)

        result = judge.evaluate_plan(
            patient_context={"chief_complaint": "fever"},
            clinical_plan="Start antibiotics.",
        )

        assert result["score"] == 0
        assert "error" in result

    def test_extract_score_parses_fraction(self):
        """_extract_score should parse 'Score: 4/5' format."""
        from evaluation.llm_judge import LLMJudge

        judge = LLMJudge(llm=MagicMock())
        assert judge._extract_score("Score: 4/5\nRationale: Good.") == 4

    def test_extract_score_parses_plain_number(self):
        """_extract_score should parse plain numbers like 'Score: 3'."""
        from evaluation.llm_judge import LLMJudge

        judge = LLMJudge(llm=MagicMock())
        assert judge._extract_score("Score: 3\nRationale: OK.") == 3

    def test_extract_score_returns_zero_on_parse_failure(self):
        """_extract_score should return 0 if no score pattern is found."""
        from evaluation.llm_judge import LLMJudge

        judge = LLMJudge(llm=MagicMock())
        assert judge._extract_score("No score here.") == 0


class TestGenerateJudgingPrompt:
    """Tests for the judging prompt generation."""

    def test_import(self):
        """generate_judging_prompt should be importable."""
        from evaluation.llm_judge import generate_judging_prompt

    def test_prompt_includes_patient_context(self):
        """The judging prompt should include the patient context."""
        from evaluation.llm_judge import generate_judging_prompt

        prompt = generate_judging_prompt(
            patient_context={"chief_complaint": "chest pain", "age": "55"},
            clinical_plan="Admit to CCU. Start heparin.",
        )
        assert "chest pain" in prompt
        assert "55" in prompt

    def test_prompt_includes_clinical_plan(self):
        """The judging prompt should include the clinical plan."""
        from evaluation.llm_judge import generate_judging_prompt

        prompt = generate_judging_prompt(
            patient_context={"chief_complaint": "fever"},
            clinical_plan="Start IV antibiotics. Blood cultures x2.",
        )
        assert "antibiotics" in prompt

    def test_prompt_asks_for_score(self):
        """The judging prompt should ask the LLM to provide a score."""
        from evaluation.llm_judge import generate_judging_prompt

        prompt = generate_judging_prompt(
            patient_context={},
            clinical_plan="Some plan.",
        )
        assert "score" in prompt.lower() or "rate" in prompt.lower()
