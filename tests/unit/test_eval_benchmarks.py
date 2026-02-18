"""Tests for benchmark data loaders — Phase 10.1

All dataset loading is mocked (no network calls).
Tests verify: load functions, prompt formatting, specialty filtering.
"""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset


# ---------------------------------------------------------------------------
# Fixtures: fake HuggingFace datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_medqa_rows():
    """Minimal rows matching GBaker/MedQA-USMLE-4-options schema."""
    return [
        {
            "question": "A 55-year-old man presents with chest pain. Which is most likely?",
            "options": {"A": "GERD", "B": "MI", "C": "PE", "D": "Costochondritis"},
            "answer_idx": "B",
            "meta_info": "step1",
        },
        {
            "question": "A child has a barking cough. Diagnosis?",
            "options": {"A": "Croup", "B": "Epiglottitis", "C": "Asthma", "D": "Bronchiolitis"},
            "answer_idx": "A",
            "meta_info": "step2",
        },
    ]


@pytest.fixture
def fake_pubmedqa_rows():
    """Minimal rows matching qiaojin/PubMedQA pqa_labeled schema."""
    return [
        {
            "pubid": 12345678,
            "question": "Does metformin reduce mortality in type 2 diabetes?",
            "context": {"contexts": ["Study showed significant reduction..."]},
            "long_answer": "Yes, metformin significantly reduces mortality.",
            "final_decision": "yes",
        },
        {
            "pubid": 87654321,
            "question": "Is ibuprofen effective for migraines?",
            "context": {"contexts": ["A randomized trial demonstrated..."]},
            "long_answer": "Results were inconclusive.",
            "final_decision": "maybe",
        },
    ]


@pytest.fixture
def fake_medmcqa_rows():
    """Minimal rows matching openlifescienceai/medmcqa schema."""
    return [
        {
            "question": "Which enzyme is deficient in Gaucher disease?",
            "opa": "Glucocerebrosidase",
            "opb": "Sphingomyelinase",
            "opc": "Hexosaminidase A",
            "opd": "Alpha-galactosidase",
            "cop": 0,  # correct option index (0=A)
            "subject_name": "Biochemistry",
            "topic_name": "Lysosomal Storage Diseases",
        },
        {
            "question": "Most common cause of nephrotic syndrome in children?",
            "opa": "Membranous nephropathy",
            "opb": "Minimal change disease",
            "opc": "FSGS",
            "opd": "IgA nephropathy",
            "cop": 1,  # correct option index (1=B)
            "subject_name": "Pediatrics",
            "topic_name": "Nephrology",
        },
    ]


# ---------------------------------------------------------------------------
# Test: load_medqa
# ---------------------------------------------------------------------------

class TestLoadMedqa:
    """Tests for load_medqa() data loader."""

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medqa_returns_list_of_dicts(self, mock_load, fake_medqa_rows):
        mock_load.return_value = {"test": Dataset.from_list(fake_medqa_rows)}
        from evaluation.benchmarks import load_medqa

        result = load_medqa()
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medqa_has_required_keys(self, mock_load, fake_medqa_rows):
        mock_load.return_value = {"test": Dataset.from_list(fake_medqa_rows)}
        from evaluation.benchmarks import load_medqa

        result = load_medqa()
        for item in result:
            assert "question" in item
            assert "options" in item
            assert "answer" in item

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medqa_respects_limit(self, mock_load, fake_medqa_rows):
        mock_load.return_value = {"test": Dataset.from_list(fake_medqa_rows)}
        from evaluation.benchmarks import load_medqa

        result = load_medqa(limit=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: load_pubmedqa
# ---------------------------------------------------------------------------

class TestLoadPubmedqa:
    """Tests for load_pubmedqa() data loader."""

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_pubmedqa_returns_list_of_dicts(self, mock_load, fake_pubmedqa_rows):
        mock_load.return_value = {"train": Dataset.from_list(fake_pubmedqa_rows)}
        from evaluation.benchmarks import load_pubmedqa

        result = load_pubmedqa()
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_pubmedqa_has_required_keys(self, mock_load, fake_pubmedqa_rows):
        mock_load.return_value = {"train": Dataset.from_list(fake_pubmedqa_rows)}
        from evaluation.benchmarks import load_pubmedqa

        result = load_pubmedqa()
        for item in result:
            assert "question" in item
            assert "context" in item
            assert "answer" in item

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_pubmedqa_answer_is_yes_no_maybe(self, mock_load, fake_pubmedqa_rows):
        mock_load.return_value = {"train": Dataset.from_list(fake_pubmedqa_rows)}
        from evaluation.benchmarks import load_pubmedqa

        result = load_pubmedqa()
        for item in result:
            assert item["answer"] in ("yes", "no", "maybe")


# ---------------------------------------------------------------------------
# Test: load_medmcqa
# ---------------------------------------------------------------------------

class TestLoadMedmcqa:
    """Tests for load_medmcqa() data loader."""

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medmcqa_returns_list_of_dicts(self, mock_load, fake_medmcqa_rows):
        mock_load.return_value = {"validation": Dataset.from_list(fake_medmcqa_rows)}
        from evaluation.benchmarks import load_medmcqa

        result = load_medmcqa()
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medmcqa_has_required_keys(self, mock_load, fake_medmcqa_rows):
        mock_load.return_value = {"validation": Dataset.from_list(fake_medmcqa_rows)}
        from evaluation.benchmarks import load_medmcqa

        result = load_medmcqa()
        for item in result:
            assert "question" in item
            assert "options" in item
            assert "answer" in item
            assert "subject" in item

    @patch("evaluation.benchmarks.hf_load_dataset")
    def test_load_medmcqa_answer_is_letter(self, mock_load, fake_medmcqa_rows):
        mock_load.return_value = {"validation": Dataset.from_list(fake_medmcqa_rows)}
        from evaluation.benchmarks import load_medmcqa

        result = load_medmcqa()
        for item in result:
            assert item["answer"] in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Test: format_medqa_prompt
# ---------------------------------------------------------------------------

class TestFormatMedqaPrompt:
    """Tests for prompt formatting utilities."""

    def test_format_medqa_prompt_contains_question(self):
        from evaluation.benchmarks import format_medqa_prompt

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "Flu", "B": "Cold", "C": "COVID", "D": "Allergy"},
        }
        prompt = format_medqa_prompt(item)
        assert "What is the diagnosis?" in prompt

    def test_format_medqa_prompt_contains_all_options(self):
        from evaluation.benchmarks import format_medqa_prompt

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "Flu", "B": "Cold", "C": "COVID", "D": "Allergy"},
        }
        prompt = format_medqa_prompt(item)
        assert "A)" in prompt or "A." in prompt or "(A)" in prompt
        assert "Flu" in prompt
        assert "Cold" in prompt
        assert "COVID" in prompt
        assert "Allergy" in prompt

    def test_format_medqa_prompt_asks_for_single_letter(self):
        from evaluation.benchmarks import format_medqa_prompt

        item = {
            "question": "What is the diagnosis?",
            "options": {"A": "Flu", "B": "Cold", "C": "COVID", "D": "Allergy"},
        }
        prompt = format_medqa_prompt(item)
        # prompt should instruct the model to answer with a single letter
        assert "letter" in prompt.lower() or "answer" in prompt.lower()


# ---------------------------------------------------------------------------
# Test: filter_by_specialty
# ---------------------------------------------------------------------------

class TestFilterBySpecialty:
    """Tests for specialty-based filtering (MedMCQA subjects)."""

    def test_filter_by_specialty_returns_matching(self):
        from evaluation.benchmarks import filter_by_specialty

        items = [
            {"question": "Q1", "subject": "Pediatrics"},
            {"question": "Q2", "subject": "Biochemistry"},
            {"question": "Q3", "subject": "Pediatrics"},
        ]
        result = filter_by_specialty(items, "Pediatrics")
        assert len(result) == 2
        assert all(r["subject"] == "Pediatrics" for r in result)

    def test_filter_by_specialty_case_insensitive(self):
        from evaluation.benchmarks import filter_by_specialty

        items = [
            {"question": "Q1", "subject": "pediatrics"},
            {"question": "Q2", "subject": "PEDIATRICS"},
        ]
        result = filter_by_specialty(items, "Pediatrics")
        assert len(result) == 2

    def test_filter_by_specialty_no_match_returns_empty(self):
        from evaluation.benchmarks import filter_by_specialty

        items = [{"question": "Q1", "subject": "Biochemistry"}]
        result = filter_by_specialty(items, "Cardiology")
        assert len(result) == 0
