"""
MedGemma-Council Gradio Application.

Provides a Gradio-based clinical decision support interface with:
- Patient context form
- Medical image upload
- Council execution and visualization
- Safety guardrails on all displayed output

Designed for Kaggle compatibility (Gradio works natively in Kaggle notebooks).

Run with: python app_gradio.py
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

from graph import build_council_graph
from utils.safety import add_disclaimer, redact_pii, scan_for_red_flags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backing logic (tested without Gradio runtime)
# ---------------------------------------------------------------------------


def create_initial_state() -> Dict[str, Any]:
    """Create a blank CouncilState for a new session."""
    return {
        "messages": [],
        "patient_context": {},
        "medical_images": [],
        "agent_outputs": {},
        "debate_history": [],
        "consensus_reached": False,
        "research_findings": "",
        "conflict_detected": False,
        "iteration_count": 0,
        "final_plan": "",
    }


def build_patient_context(
    age: int,
    sex: str,
    chief_complaint: str,
    history: str = "",
    medications: str = "",
) -> Dict[str, Any]:
    """
    Build a patient_context dict from form inputs.

    Args:
        age: Patient age in years.
        sex: Patient sex.
        chief_complaint: Primary reason for visit.
        history: Past medical history (free text).
        medications: Comma-separated medication list.

    Returns:
        Structured patient context dictionary.
    """
    med_list: List[str] = []
    if medications:
        med_list = [m.strip() for m in medications.split(",") if m.strip()]

    ctx: Dict[str, Any] = {
        "age": age,
        "sex": sex,
        "chief_complaint": chief_complaint,
        "history": history,
        "medications": med_list,
    }

    return ctx


def process_uploaded_images(
    files: Optional[List[str]],
) -> List[str]:
    """
    Process Gradio file uploads and return their paths.

    Gradio file uploads provide file paths (strings) directly,
    unlike Streamlit which gives file-like objects.

    Args:
        files: List of file paths from Gradio file upload, or None.

    Returns:
        List of valid file paths.
    """
    if not files:
        return []

    paths = []
    for f in files:
        if isinstance(f, str) and os.path.exists(f):
            paths.append(f)
        elif hasattr(f, "name"):
            # Handle Gradio NamedString or UploadedFile-like objects
            paths.append(f.name)

    return paths


def format_plan_output(plan: str) -> str:
    """
    Format and safety-check the final clinical plan.

    Applies:
    1. PII redaction
    2. Red flag scanning with emergency override
    3. Disclaimer

    Args:
        plan: Raw final plan text.

    Returns:
        Safety-checked formatted plan string.
    """
    safe_plan = redact_pii(plan)

    sections = []

    # Red flag check
    flags = scan_for_red_flags(safe_plan)
    if flags.get("flagged"):
        sections.append(f"{'='*60}")
        sections.append("**EMERGENCY ALERT**")
        sections.append(flags["emergency_message"])
        sections.append(f"{'='*60}\n")

    sections.append(safe_plan)

    output = "\n".join(sections)
    output = add_disclaimer(output)

    return output


def format_specialists_output(agent_outputs: Dict[str, str]) -> str:
    """
    Format specialist findings for display.

    Args:
        agent_outputs: Dict of agent name -> finding text.

    Returns:
        Formatted specialist findings string.
    """
    if not agent_outputs:
        return ""

    sections = []
    for agent_name, finding in agent_outputs.items():
        safe_finding = redact_pii(finding)
        sections.append(f"### {agent_name}\n{safe_finding}")

    return "\n\n".join(sections)


def run_council_analysis(
    age: int,
    sex: str,
    chief_complaint: str,
    history: str = "",
    medications: str = "",
    images: Optional[List[str]] = None,
    progress: Optional[gr.Progress] = None,
) -> Tuple[str, str, str]:
    """
    Run the council analysis and return formatted results.

    This is the main callback wired to the Gradio submit button.

    Args:
        age: Patient age.
        sex: Patient sex.
        chief_complaint: Primary complaint.
        history: Past medical history.
        medications: Comma-separated medication string.
        images: Optional list of image file paths from Gradio upload.
        progress: Gradio progress tracker for UI feedback.

    Returns:
        Tuple of (plan_output, specialists_output, status_output).
    """
    def update_progress(p: float, desc: str):
        if progress:
            progress(p, desc=desc)

    update_progress(0, "Validating input...")

    if not chief_complaint or not chief_complaint.strip():
        return (
            "Please enter a chief complaint to proceed.",
            "",
            "Status: Awaiting input",
        )

    update_progress(0.1, "Building patient context...")
    patient_ctx = build_patient_context(
        age=age,
        sex=sex,
        chief_complaint=chief_complaint,
        history=history,
        medications=medications,
    )

    update_progress(0.15, "Processing images...")
    image_paths = process_uploaded_images(images)

    update_progress(0.2, "Initializing council graph...")
    state = create_initial_state()
    state["messages"] = [{"role": "user", "content": chief_complaint}]
    state["patient_context"] = patient_ctx
    state["medical_images"] = image_paths

    update_progress(0.25, "Running specialist analysis...")
    try:
        graph = build_council_graph()
        update_progress(0.4, "Council debating clinical findings...")
        result = graph.invoke(state)
        update_progress(0.9, "Synthesizing final plan...")
    except Exception as e:
        logger.error(f"Council execution failed: {e}")
        return (
            f"Error: Council execution failed — {str(e)}",
            "",
            "Status: Error",
        )

    plan = format_plan_output(result.get("final_plan", ""))
    specialists = format_specialists_output(result.get("agent_outputs", {}))

    consensus = "Yes" if result.get("consensus_reached") else "No"
    conflict = "Yes" if result.get("conflict_detected") else "No"
    rounds = result.get("iteration_count", 0)
    n_specialists = len(result.get("agent_outputs", {}))

    status = (
        f"**Consensus:** {consensus}\n"
        f"**Conflict Detected:** {conflict}\n"
        f"**Debate Rounds:** {rounds}\n"
        f"**Specialists Consulted:** {n_specialists}"
    )

    update_progress(1.0, "Complete!")
    return plan, specialists, status


# ---------------------------------------------------------------------------
# Gradio UI builder
# ---------------------------------------------------------------------------


def build_gradio_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks application.

    Returns:
        A gr.Blocks instance ready to be launched.
    """
    custom_css = """
    .plan-output {
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
    }
    .specialists-output {
        max-height: 300px;
        overflow-y: auto;
    }
    .status-output {
        margin-top: 10px;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(
        title="MedGemma Council",
        css=custom_css,
    ) as app:
        gr.Markdown(
            "# MedGemma Council of Experts\n"
            "Multi-agent clinical decision support powered by MedGemma 1.5\n\n"
            "> **Disclaimer:** This is an AI research project. It is NOT a "
            "substitute for professional medical advice, diagnosis, or treatment."
        )

        with gr.Row():
            # --- Left column: Patient form ---
            with gr.Column(scale=2):
                gr.Markdown("## Patient Information")

                with gr.Row():
                    age_input = gr.Number(
                        label="Age",
                        value=50,
                        minimum=0,
                        maximum=120,
                        precision=0,
                    )
                    sex_input = gr.Dropdown(
                        label="Sex",
                        choices=["Male", "Female", "Other"],
                        value="Male",
                    )

                complaint_input = gr.Textbox(
                    label="Chief Complaint",
                    placeholder="e.g., Chest pain radiating to left arm",
                    lines=2,
                )
                history_input = gr.Textbox(
                    label="Past Medical History",
                    placeholder="e.g., Hypertension, Type 2 Diabetes",
                    lines=2,
                )
                medications_input = gr.Textbox(
                    label="Current Medications (comma-separated)",
                    placeholder="e.g., Aspirin, Lisinopril, Metformin",
                )
                images_input = gr.File(
                    label="Medical Images (X-ray, CT, MRI)",
                    file_types=["image", ".dcm"],
                    file_count="multiple",
                )

                submit_btn = gr.Button(
                    "Run Council Analysis",
                    variant="primary",
                )

            # --- Right column: Results ---
            with gr.Column(scale=3):
                gr.Markdown("## Results")

                plan_output = gr.Markdown(
                    label="Clinical Management Plan",
                    value="*Submit a case to start the council analysis.*",
                    elem_classes=["plan-output"],
                )

                status_output = gr.Markdown(
                    label="Council Status",
                    value="",
                    elem_classes=["status-output"],
                )

                with gr.Accordion("Specialist Findings", open=False):
                    specialists_output = gr.Markdown(
                        label="Specialist Findings",
                        value="",
                        elem_classes=["specialists-output"],
                    )

        # --- Wire up the submit button ---
        submit_btn.click(
            fn=run_council_analysis,
            inputs=[
                age_input,
                sex_input,
                complaint_input,
                history_input,
                medications_input,
                images_input,
            ],
            outputs=[plan_output, specialists_output, status_output],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Launch the Gradio application."""
    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
