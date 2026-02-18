"""CLI runner for the clinical benchmark evaluation harness.

Usage:
    python -m src.evaluation.runner --benchmark medqa --limit 100 --output results.json
    python -m src.evaluation.runner --benchmark medmcqa --specialty Pediatrics --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from evaluation.benchmarks import (
    load_medqa,
    load_pubmedqa,
    load_medmcqa,
    filter_by_specialty,
)
from evaluation.evaluator import CouncilEvaluator
from evaluation.metrics import generate_report


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_BENCHMARKS = ("medqa", "pubmedqa", "medmcqa")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run MedGemma-Council on clinical benchmarks",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=_BENCHMARKS,
        default="medqa",
        help="Benchmark dataset to evaluate on (default: medqa)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (default: stdout only)",
    )
    parser.add_argument(
        "--specialty",
        type=str,
        default=None,
        help="Filter MedMCQA by specialty/subject name",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------

def run_evaluation(
    benchmark: str = "medqa",
    limit: int | None = None,
    output: str | None = None,
    specialty: str | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Run evaluation on the specified benchmark.

    Args:
        benchmark: One of 'medqa', 'pubmedqa', 'medmcqa'.
        limit: Max items to evaluate.
        output: Path to write JSON results.
        specialty: Filter MedMCQA items by subject.
        graph: Optional pre-built council graph (for testing).

    Returns dict with 'report' and 'results' keys.
    """
    # Load dataset — use if/elif so patches on module-level names work in tests
    if benchmark == "medqa":
        items = load_medqa(limit=limit)
    elif benchmark == "pubmedqa":
        items = load_pubmedqa(limit=limit)
    elif benchmark == "medmcqa":
        items = load_medmcqa(limit=limit)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Optional specialty filter (MedMCQA only)
    if specialty and benchmark == "medmcqa":
        items = filter_by_specialty(items, specialty)

    # Evaluate
    evaluator = CouncilEvaluator(graph=graph)
    results = evaluator.evaluate_batch(items)

    # Propagate subject info from items to results (for per-specialty metrics)
    for item, result in zip(items, results):
        if "subject" in item:
            result["subject"] = item["subject"]

    # Generate report
    report = generate_report(results, benchmark_name=benchmark)

    output_data = {"report": report, "results": results}

    # Write to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)

    return output_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    result = run_evaluation(
        benchmark=args.benchmark,
        limit=args.limit,
        output=args.output,
        specialty=args.specialty,
    )

    # Print summary to stdout
    report = result["report"]
    print(f"\n{'='*60}")
    print(f"Benchmark: {report['benchmark']}")
    print(f"Total:     {report['total']}")
    print(f"Correct:   {report['correct_count']}")
    print(f"Accuracy:  {report['accuracy']:.1%}")
    print(f"95% CI:    [{report['ci_lower']:.3f}, {report['ci_upper']:.3f}]")

    if "per_specialty" in report:
        print(f"\nPer-specialty breakdown:")
        for specialty, stats in sorted(report["per_specialty"].items()):
            print(f"  {specialty:30s}  {stats['accuracy']:.1%}  (n={stats['total']})")

    print(f"{'='*60}\n")

    if args.output:
        print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
