"""
MedGemma-Council CLI Interface.

Provides a command-line and programmatic interface for running the
council without any web framework dependency. Designed for:
- Kaggle notebook usage (import and call directly)
- Local terminal usage (argparse CLI)
- Testing and debugging

Usage (CLI):
    python council_cli.py --age 65 --sex Male --complaint "Chest pain"

Usage (Python):
    from council_cli import run_council_cli
    result = run_council_cli(age=65, sex="Male", chief_complaint="Chest pain")
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graph import build_council_graph
from utils.safety import add_disclaimer, redact_pii, scan_for_red_flags

logger = logging.getLogger(__name__)


def build_state(
    age: int,
    sex: str,
    chief_complaint: str,
    history: str = "",
    medications: Optional[List[str]] = None,
    vitals: Optional[Dict[str, Any]] = None,
    labs: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a valid CouncilState from patient parameters.

    Args:
        age: Patient age in years.
        sex: Patient sex.
        chief_complaint: Primary reason for visit.
        history: Past medical history (free text).
        medications: List of current medications.
        vitals: Dict of vital signs.
        labs: Dict of lab values.
        image_paths: List of file paths to medical images.

    Returns:
        A valid CouncilState dictionary ready for graph invocation.
    """
    patient_context: Dict[str, Any] = {
        "age": age,
        "sex": sex,
        "chief_complaint": chief_complaint,
        "history": history,
        "medications": medications or [],
    }
    if vitals:
        patient_context["vitals"] = vitals
    if labs:
        patient_context["labs"] = labs

    return {
        "messages": [{"role": "user", "content": chief_complaint}],
        "patient_context": patient_context,
        "medical_images": image_paths or [],
        "agent_outputs": {},
        "debate_history": [],
        "consensus_reached": False,
        "research_findings": "",
        "conflict_detected": False,
        "iteration_count": 0,
        "final_plan": "",
    }


def run_council_cli(
    age: int,
    sex: str,
    chief_complaint: str,
    history: str = "",
    medications: Optional[List[str]] = None,
    vitals: Optional[Dict[str, Any]] = None,
    labs: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    verbose: bool = True,
    text_model_id: Optional[str] = None,
    clear_model_cache: bool = False,
) -> Dict[str, Any]:
    """
    Run the MedGemma Council and return the full result state.

    This is the primary programmatic entry point for Kaggle notebooks
    and Python scripts.

    Args:
        age: Patient age in years.
        sex: Patient sex.
        chief_complaint: Primary reason for visit.
        history: Past medical history.
        medications: List of current medications.
        vitals: Dict of vital signs.
        labs: Dict of lab values.
        image_paths: List of paths to medical images.
        verbose: If True (default), configure logging to DEBUG level
            for maximum visibility into model inference. If False,
            set logging to WARNING.
        text_model_id: Optional HuggingFace model ID to override the
            default text model (google/medgemma-4b-it). Sets
            the MEDGEMMA_TEXT_MODEL_ID env var for ModelFactory.
            Use "google/medgemma-27b-text-it" for optional larger-model
            inference on stronger hardware.
        clear_model_cache: If True, clear the model cache before running.
            Use this if you see CUDA errors (CUBLAS_STATUS_ALLOC_FAILED)
            which indicate corrupted GPU state from a previous run.

    Returns:
        The final CouncilState dict after graph execution.
    """
    # Configure logging based on verbose flag
    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
        logger.debug("Verbose mode enabled — logging at DEBUG level")
    else:
        logging.basicConfig(level=logging.WARNING, force=True)

    # Clear model cache if requested (for recovering from CUDA errors)
    if clear_model_cache:
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()
        logger.info("Model cache cleared")
        
        # Also try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass

    # Set text model override env var if provided
    _env_was_set = False
    if text_model_id is not None:
        os.environ["MEDGEMMA_TEXT_MODEL_ID"] = text_model_id
        _env_was_set = True
        logger.info(f"Text model override: {text_model_id}")

    state = build_state(
        age=age,
        sex=sex,
        chief_complaint=chief_complaint,
        history=history,
        medications=medications,
        vitals=vitals,
        labs=labs,
        image_paths=image_paths,
    )

    try:
        graph = build_council_graph()
        result = graph.invoke(state)
        return result
    except Exception as e:
        logger.error(f"Council execution failed: {e}")
        return {
            **state,
            "final_plan": f"Error: Council execution failed — {str(e)}",
            "consensus_reached": False,
        }
    finally:
        # Clean up env var to avoid leaking into subsequent calls
        if _env_was_set:
            os.environ.pop("MEDGEMMA_TEXT_MODEL_ID", None)
            logger.debug("Cleaned up MEDGEMMA_TEXT_MODEL_ID env var")


def format_result(result: Dict[str, Any], output_format: str = "text") -> str:
    """
    Format the council result for display.

    Args:
        result: The CouncilState dict returned by run_council_cli.
        output_format: 'text' for human-readable, 'json' for machine-readable.

    Returns:
        Formatted string.
    """
    if output_format == "json":
        # Serialize with safety applied
        safe_result = {
            "final_plan": redact_pii(result.get("final_plan", "")),
            "agent_outputs": {
                k: redact_pii(v) for k, v in result.get("agent_outputs", {}).items()
            },
            "debate_history": [
                redact_pii(entry) for entry in result.get("debate_history", [])
            ],
            "consensus_reached": result.get("consensus_reached", False),
            "conflict_detected": result.get("conflict_detected", False),
            "iteration_count": result.get("iteration_count", 0),
            "research_findings": redact_pii(result.get("research_findings", "")),
        }
        return json.dumps(safe_result, indent=2)

    # Text format
    sections = []
    final_plan = redact_pii(result.get("final_plan", ""))

    # Red flag check
    flags = scan_for_red_flags(final_plan)
    if flags.get("flagged"):
        sections.append(f"{'='*60}")
        sections.append("EMERGENCY ALERT")
        sections.append(flags["emergency_message"])
        sections.append(f"{'='*60}\n")

    sections.append(f"{'='*60}")
    sections.append("CLINICAL MANAGEMENT PLAN")
    sections.append(f"{'='*60}")
    sections.append(final_plan)
    sections.append("")

    # Specialist findings
    agent_outputs = result.get("agent_outputs", {})
    if agent_outputs:
        sections.append(f"{'-'*60}")
        sections.append("SPECIALIST FINDINGS")
        sections.append(f"{'-'*60}")
        for agent_name, finding in agent_outputs.items():
            safe_finding = redact_pii(finding)
            sections.append(f"\n[{agent_name}]")
            sections.append(safe_finding)

    # Debate history
    debate = result.get("debate_history", [])
    if debate:
        sections.append(f"\n{'-'*60}")
        sections.append("DEBATE HISTORY")
        sections.append(f"{'-'*60}")
        for i, entry in enumerate(debate, 1):
            sections.append(f"\nRound {i}: {redact_pii(entry)}")

    # Status
    sections.append(f"\n{'-'*60}")
    sections.append("STATUS")
    sections.append(f"{'-'*60}")
    sections.append(f"Consensus: {'Yes' if result.get('consensus_reached') else 'No'}")
    sections.append(f"Conflict: {'Yes' if result.get('conflict_detected') else 'No'}")
    sections.append(f"Debate Rounds: {result.get('iteration_count', 0)}")

    # Disclaimer
    output = "\n".join(sections)
    output = add_disclaimer(output)

    return output


def main():
    """CLI entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="MedGemma-Council: Multi-Agent Clinical Decision Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python council_cli.py --age 65 --sex Male --complaint 'Chest pain'\n"
            "  python council_cli.py --age 5 --sex Female --complaint 'Fever and cough' --output json\n"
            "  python council_cli.py --age 45 --sex Male --complaint 'Skin lesion' --images img1.png img2.png\n"
            "  python council_cli.py --age 65 --sex Male --complaint 'Chest pain' --model-id google/medgemma-27b-text-it\n"
            "  python council_cli.py --age 65 --sex Male --complaint 'Chest pain' --quiet\n"
        ),
    )
    parser.add_argument("--age", type=int, required=True, help="Patient age in years")
    parser.add_argument("--sex", type=str, required=True, choices=["Male", "Female", "Other"], help="Patient sex")
    parser.add_argument("--complaint", type=str, required=True, help="Chief complaint")
    parser.add_argument("--history", type=str, default="", help="Past medical history")
    parser.add_argument("--medications", type=str, nargs="*", default=[], help="Current medications")
    parser.add_argument("--images", type=str, nargs="*", default=[], help="Paths to medical images")
    parser.add_argument("--output", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="HuggingFace model ID for text inference (default: google/medgemma-4b-it). "
             "Optional override: google/medgemma-27b-text-it on larger hardware.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce logging to WARNING level (default is verbose/DEBUG)",
    )

    args = parser.parse_args()

    result = run_council_cli(
        age=args.age,
        sex=args.sex,
        chief_complaint=args.complaint,
        history=args.history,
        medications=args.medications,
        image_paths=args.images,
        verbose=not args.quiet,
        text_model_id=args.model_id,
    )

    print(format_result(result, output_format=args.output))


if __name__ == "__main__":
    main()
