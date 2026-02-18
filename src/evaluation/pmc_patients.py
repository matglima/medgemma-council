"""
PMC-Patients dataset loader and prompt formatter.

Loads patient case descriptions from the PMC-Patients dataset on HuggingFace
and formats them into prompts for clinical plan evaluation.
"""

from typing import Dict, List, Optional

from datasets import load_dataset as hf_load_dataset


def load_pmc_patients(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load PMC-Patients dataset from HuggingFace.

    Args:
        limit: Maximum number of items to return. None returns all.

    Returns:
        List of dicts with keys: patient_id, patient_text, title.
    """
    dataset = hf_load_dataset("zhengyun21/PMC-Patients")
    records = dataset["train"]

    items = []
    for record in records:
        items.append({
            "patient_id": record["patient_id"],
            "patient_text": record["patient"],
            "title": record["title"],
        })
        if limit is not None and len(items) >= limit:
            break

    return items


def format_pmc_patient_prompt(item: Dict[str, str]) -> str:
    """Format a PMC-Patients item into a clinical evaluation prompt.

    Args:
        item: Dict with patient_id, patient_text, and title.

    Returns:
        Formatted prompt string for clinical plan generation.
    """
    return (
        f"You are a clinical expert. Read the following patient description and "
        f"provide a comprehensive clinical plan including diagnosis, workup, and "
        f"treatment.\n\n"
        f"Patient Description:\n{item['patient_text']}\n\n"
        f"Provide your clinical plan below:"
    )
