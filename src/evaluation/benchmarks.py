"""Benchmark data loaders for MedQA, PubMedQA, and MedMCQA.

Each loader downloads the dataset via HuggingFace `datasets`, normalises rows
into a common schema, and returns a plain list[dict].

Network access is only needed on the first call (datasets caches locally).
In tests, `hf_load_dataset` is mocked so no network calls occur.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset as hf_load_dataset


# ---------------------------------------------------------------------------
# MedQA  (GBaker/MedQA-USMLE-4-options)
# ---------------------------------------------------------------------------

def load_medqa(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the MedQA-USMLE 4-option test split.

    Returns list of dicts with keys: question, options (dict A-D), answer (letter).
    """
    ds = hf_load_dataset("GBaker/MedQA-USMLE-4-options")
    rows = ds["test"]
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append({
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer_idx"],
        })
        if limit is not None and len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# PubMedQA  (qiaojin/PubMedQA, pqa_labeled)
# ---------------------------------------------------------------------------

def load_pubmedqa(limit: int | None = None) -> list[dict[str, Any]]:
    """Load the PubMedQA labelled split.

    Returns list of dicts with keys: question, context (str), answer (yes/no/maybe).
    """
    ds = hf_load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    rows = ds["train"]  # pqa_labeled only has a 'train' split
    results: list[dict[str, Any]] = []
    for row in rows:
        # context is a dict with a "contexts" key containing a list of strings
        ctx = row.get("context", {})
        context_str = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
        results.append({
            "question": row["question"],
            "context": context_str,
            "answer": row["final_decision"],
        })
        if limit is not None and len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# MedMCQA  (openlifescienceai/medmcqa)
# ---------------------------------------------------------------------------

_COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_medmcqa(limit: int | None = None) -> list[dict[str, Any]]:
    """Load MedMCQA validation split.

    Returns list of dicts with keys: question, options (dict A-D), answer (letter),
    subject (str).
    """
    ds = hf_load_dataset("openlifescienceai/medmcqa")
    rows = ds["validation"]
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append({
            "question": row["question"],
            "options": {
                "A": row["opa"],
                "B": row["opb"],
                "C": row["opc"],
                "D": row["opd"],
            },
            "answer": _COP_TO_LETTER.get(row["cop"], "A"),
            "subject": row.get("subject_name", "Unknown"),
        })
        if limit is not None and len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_medqa_prompt(item: dict[str, Any]) -> str:
    """Format a MedQA/MedMCQA item into a multiple-choice prompt string.

    The prompt instructs the model to answer with a single letter (A-D).
    """
    lines = [item["question"], ""]
    for letter, text in item["options"].items():
        lines.append(f"({letter}) {text}")
    lines.append("")
    lines.append("Answer with only the single letter of the correct option.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Specialty filtering (MedMCQA subjects)
# ---------------------------------------------------------------------------

def filter_by_specialty(
    items: list[dict[str, Any]],
    specialty: str,
) -> list[dict[str, Any]]:
    """Filter items by subject/specialty name (case-insensitive)."""
    target = specialty.lower()
    return [item for item in items if item.get("subject", "").lower() == target]
