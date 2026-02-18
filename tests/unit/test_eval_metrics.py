"""Tests for evaluation metrics — Phase 10.6

Tests verify: compute_accuracy, compute_per_specialty_accuracy, generate_report.
"""

import pytest


# ---------------------------------------------------------------------------
# Test: compute_accuracy
# ---------------------------------------------------------------------------

class TestComputeAccuracy:
    """Tests for overall accuracy computation."""

    def test_perfect_score(self):
        from evaluation.metrics import compute_accuracy

        results = [
            {"is_correct": True},
            {"is_correct": True},
            {"is_correct": True},
        ]
        acc = compute_accuracy(results)
        assert acc == pytest.approx(1.0)

    def test_zero_score(self):
        from evaluation.metrics import compute_accuracy

        results = [
            {"is_correct": False},
            {"is_correct": False},
        ]
        acc = compute_accuracy(results)
        assert acc == pytest.approx(0.0)

    def test_partial_score(self):
        from evaluation.metrics import compute_accuracy

        results = [
            {"is_correct": True},
            {"is_correct": False},
            {"is_correct": True},
            {"is_correct": False},
        ]
        acc = compute_accuracy(results)
        assert acc == pytest.approx(0.5)

    def test_empty_results_returns_zero(self):
        from evaluation.metrics import compute_accuracy

        acc = compute_accuracy([])
        assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: compute_per_specialty_accuracy
# ---------------------------------------------------------------------------

class TestComputePerSpecialtyAccuracy:
    """Tests for per-specialty (subject) accuracy breakdown."""

    def test_returns_dict_of_specialties(self):
        from evaluation.metrics import compute_per_specialty_accuracy

        results = [
            {"is_correct": True, "subject": "Pediatrics"},
            {"is_correct": False, "subject": "Biochemistry"},
            {"is_correct": True, "subject": "Pediatrics"},
        ]
        breakdown = compute_per_specialty_accuracy(results)
        assert isinstance(breakdown, dict)
        assert "Pediatrics" in breakdown
        assert "Biochemistry" in breakdown

    def test_accuracy_per_specialty(self):
        from evaluation.metrics import compute_per_specialty_accuracy

        results = [
            {"is_correct": True, "subject": "Pediatrics"},
            {"is_correct": False, "subject": "Pediatrics"},
            {"is_correct": True, "subject": "Biochemistry"},
        ]
        breakdown = compute_per_specialty_accuracy(results)
        assert breakdown["Pediatrics"]["accuracy"] == pytest.approx(0.5)
        assert breakdown["Biochemistry"]["accuracy"] == pytest.approx(1.0)

    def test_count_per_specialty(self):
        from evaluation.metrics import compute_per_specialty_accuracy

        results = [
            {"is_correct": True, "subject": "Pediatrics"},
            {"is_correct": True, "subject": "Pediatrics"},
            {"is_correct": False, "subject": "Biochemistry"},
        ]
        breakdown = compute_per_specialty_accuracy(results)
        assert breakdown["Pediatrics"]["total"] == 2
        assert breakdown["Biochemistry"]["total"] == 1

    def test_items_without_subject_grouped_as_unknown(self):
        from evaluation.metrics import compute_per_specialty_accuracy

        results = [
            {"is_correct": True},
            {"is_correct": False},
        ]
        breakdown = compute_per_specialty_accuracy(results)
        assert "Unknown" in breakdown
        assert breakdown["Unknown"]["total"] == 2


# ---------------------------------------------------------------------------
# Test: generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Tests for the summary report generator."""

    def test_report_is_dict(self):
        from evaluation.metrics import generate_report

        results = [
            {"is_correct": True, "predicted": "B", "correct": "B"},
            {"is_correct": False, "predicted": "A", "correct": "C"},
        ]
        report = generate_report(results, benchmark_name="medqa")
        assert isinstance(report, dict)

    def test_report_contains_accuracy(self):
        from evaluation.metrics import generate_report

        results = [
            {"is_correct": True, "predicted": "B", "correct": "B"},
            {"is_correct": False, "predicted": "A", "correct": "C"},
        ]
        report = generate_report(results, benchmark_name="medqa")
        assert "accuracy" in report
        assert report["accuracy"] == pytest.approx(0.5)

    def test_report_contains_total(self):
        from evaluation.metrics import generate_report

        results = [
            {"is_correct": True, "predicted": "B", "correct": "B"},
        ]
        report = generate_report(results, benchmark_name="medqa")
        assert report["total"] == 1
        assert report["correct_count"] == 1

    def test_report_contains_benchmark_name(self):
        from evaluation.metrics import generate_report

        results = [{"is_correct": True, "predicted": "B", "correct": "B"}]
        report = generate_report(results, benchmark_name="pubmedqa")
        assert report["benchmark"] == "pubmedqa"

    def test_report_contains_confidence_interval(self):
        from evaluation.metrics import generate_report

        results = [{"is_correct": True} for _ in range(100)]
        report = generate_report(results, benchmark_name="medqa")
        assert "ci_lower" in report
        assert "ci_upper" in report
        # 100% accuracy → CI should be close to 1.0
        assert report["ci_lower"] >= 0.9
