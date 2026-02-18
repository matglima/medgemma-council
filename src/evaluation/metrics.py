"""Evaluation metrics — accuracy, per-specialty breakdown, confidence intervals.

All functions operate on lists of result dicts produced by CouncilEvaluator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def compute_accuracy(results: list[dict[str, Any]]) -> float:
    """Compute overall accuracy (fraction of is_correct == True).

    Returns 0.0 for an empty results list.
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("is_correct"))
    return correct / len(results)


def compute_per_specialty_accuracy(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute accuracy broken down by subject/specialty.

    Items without a 'subject' key are grouped under 'Unknown'.

    Returns dict mapping specialty name -> {"accuracy": float, "total": int, "correct": int}.
    """
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        specialty = r.get("subject", "Unknown")
        buckets[specialty].append(bool(r.get("is_correct")))

    breakdown: dict[str, dict[str, Any]] = {}
    for specialty, outcomes in sorted(buckets.items()):
        total = len(outcomes)
        correct = sum(outcomes)
        breakdown[specialty] = {
            "accuracy": correct / total if total else 0.0,
            "total": total,
            "correct": correct,
        }
    return breakdown


def _wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds.  For n_total == 0, returns (0.0, 0.0).
    """
    if n_total == 0:
        return 0.0, 0.0
    p_hat = n_correct / n_total
    denominator = 1 + z**2 / n_total
    centre = p_hat + z**2 / (2 * n_total)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total)
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return max(0.0, lower), min(1.0, upper)


def generate_report(
    results: list[dict[str, Any]],
    benchmark_name: str = "unknown",
) -> dict[str, Any]:
    """Generate a summary report dict from evaluation results.

    Keys: benchmark, total, correct_count, accuracy, ci_lower, ci_upper,
          per_specialty (if subject info present).
    """
    total = len(results)
    correct_count = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct_count / total if total else 0.0
    ci_lower, ci_upper = _wilson_ci(correct_count, total)

    report: dict[str, Any] = {
        "benchmark": benchmark_name,
        "total": total,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    # Add per-specialty breakdown if any item has 'subject'
    if any("subject" in r for r in results):
        report["per_specialty"] = compute_per_specialty_accuracy(results)

    return report
