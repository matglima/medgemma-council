"""CouncilEvaluator — runs the council graph on benchmark items and scores results.

The evaluator wraps the LangGraph council, feeds each benchmark question through
the pipeline, extracts the predicted answer from the synthesis output, and
compares it to the gold label.
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.benchmarks import format_medqa_prompt


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_YES_NO_MAYBE = {"yes", "no", "maybe"}

# Matches patterns like "(B)", "answer is B", "Answer: C", or standalone letter
_LETTER_PATTERN = re.compile(
    r"""
    (?:                          # optional prefix
        (?:answer\s+is\s*|answer:\s*)  # "answer is" / "answer:"
        \(?([A-D])\)?            # letter with optional parens  → group 1
    )
    |
    \(([A-D])\)                  # (B) style                    → group 2
    |
    ^([A-D])$                    # standalone letter on its own  → group 3
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)


def extract_answer_letter(text: str) -> str | None:
    """Extract the predicted answer from free-text model output.

    For MCQ benchmarks: returns a single uppercase letter (A-D) or None.
    For PubMedQA: returns 'yes', 'no', or 'maybe'.
    """
    text_lower = text.strip().lower()

    # PubMedQA: check for yes/no/maybe
    for label in _YES_NO_MAYBE:
        if text_lower == label or f"answer is {label}" in text_lower:
            return label

    # MCQ: find letter answer
    matches = list(_LETTER_PATTERN.finditer(text))
    if len(matches) == 1:
        m = matches[0]
        letter = m.group(1) or m.group(2) or m.group(3)
        return letter.upper()

    # If multiple matches, ambiguous → return None
    if len(matches) > 1:
        return None

    return None


# ---------------------------------------------------------------------------
# CouncilEvaluator
# ---------------------------------------------------------------------------

def build_council_graph():
    """Build the default council graph. Lazy import to avoid circular deps."""
    from graph import build_council_graph as _build
    return _build()


class CouncilEvaluator:
    """Runs the council on benchmark items and collects predictions."""

    def __init__(self, graph: Any | None = None):
        if graph is None:
            graph = build_council_graph()
        self.graph = graph

    def evaluate_single(self, item: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single benchmark item through the council.

        Returns dict with keys: question, predicted, correct, is_correct.
        On error: predicted=None, is_correct=False, error=<message>.
        """
        # Build prompt depending on whether item has options (MCQ) or context (PubMedQA)
        if "options" in item:
            prompt = format_medqa_prompt(item)
        else:
            # PubMedQA style
            prompt = f"{item.get('context', '')}\n\nQuestion: {item['question']}\nAnswer yes, no, or maybe."

        try:
            result = self.graph.invoke({
                "patient_context": prompt,
                "messages": [],
                "agent_outputs": {},
                "conflict_flag": False,
                "iteration_count": 0,
                "debate_history": [],
                "active_specialists": [],
                "research_findings": "",
                "final_plan": "",
                "image_paths": [],
            })
            raw_output = result.get("final_plan", "")
            predicted = extract_answer_letter(raw_output)
        except Exception as e:
            return {
                "question": item["question"],
                "predicted": None,
                "correct": item["answer"],
                "is_correct": False,
                "error": str(e),
            }

        correct = item["answer"]
        return {
            "question": item["question"],
            "predicted": predicted,
            "correct": correct,
            "is_correct": predicted == correct,
        }

    def evaluate_batch(
        self,
        items: list[dict[str, Any]],
        progress_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of benchmark items.

        Args:
            items: List of normalised benchmark items.
            progress_callback: Optional callable(current, total) for progress.

        Returns list of result dicts from evaluate_single.
        """
        results = []
        for i, item in enumerate(items):
            result = self.evaluate_single(item)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(items))
        return results
