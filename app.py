"""
MedGemma-Council Streamlit Application.

Provides a chat-based clinical decision support interface with:
- Patient context form (sidebar)
- Medical image upload
- Council execution and visualization
- Safety guardrails on all displayed output

Run with: streamlit run app.py
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from graph import CouncilState, build_council_graph
from utils.safety import add_disclaimer, redact_pii, scan_for_red_flags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backing logic (tested without Streamlit runtime)
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
    vitals: Optional[Dict[str, Any]] = None,
    labs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a patient_context dict from form inputs.

    Args:
        age: Patient age in years.
        sex: Patient sex.
        chief_complaint: Primary reason for visit.
        history: Past medical history (free text).
        medications: Comma-separated medication list.
        vitals: Optional vitals dict.
        labs: Optional labs dict.

    Returns:
        Structured patient context dictionary.
    """
    med_list = []
    if medications:
        med_list = [m.strip() for m in medications.split(",") if m.strip()]

    ctx: Dict[str, Any] = {
        "age": age,
        "sex": sex,
        "chief_complaint": chief_complaint,
        "history": history,
        "medications": med_list,
    }

    if vitals:
        ctx["vitals"] = vitals
    if labs:
        ctx["labs"] = labs

    return ctx


def process_uploaded_images(
    files: List[Any], upload_dir: str = "data/uploads"
) -> List[str]:
    """
    Save uploaded files to disk and return their paths.

    Args:
        files: List of Streamlit UploadedFile objects.
        upload_dir: Directory to save files into.

    Returns:
        List of saved file paths.
    """
    if not files:
        return []

    os.makedirs(upload_dir, exist_ok=True)
    paths = []

    for f in files:
        file_path = os.path.join(upload_dir, f.name)
        with open(file_path, "wb") as out:
            out.write(f.read())
        paths.append(file_path)

    return paths


def run_council(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the council graph on the given state.

    Args:
        state: A valid CouncilState dict.

    Returns:
        The final state after graph execution, or an error state.
    """
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


def format_council_output(
    final_plan: str,
    agent_outputs: Dict[str, str],
    debate_history: Optional[List[str]] = None,
) -> str:
    """
    Format council results for display with safety guardrails.

    Applies:
    1. PII redaction on all text.
    2. Red flag scanning with emergency override.
    3. Disclaimer appended.

    Args:
        final_plan: The synthesized clinical plan.
        agent_outputs: Dict of specialist findings.
        debate_history: Optional debate round log.

    Returns:
        Formatted, safety-checked output string.
    """
    sections = []

    # --- Final Plan ---
    safe_plan = redact_pii(final_plan)
    flags = scan_for_red_flags(safe_plan)
    if flags:
        emergency_msg = (
            "**EMERGENCY ALERT**: Red flags detected in clinical assessment. "
            "Immediate actions required:\n"
        )
        for flag in flags:
            emergency_msg += f"- {flag}\n"
        emergency_msg += (
            "\n**Call 911 / activate emergency response immediately.** "
            "This system is not a substitute for emergency medical care."
        )
        sections.append(emergency_msg)

    sections.append(f"## Clinical Plan\n\n{safe_plan}")

    # --- Specialist Findings ---
    if agent_outputs:
        sections.append("## Specialist Findings\n")
        for agent_name, finding in agent_outputs.items():
            safe_finding = redact_pii(finding)
            sections.append(f"### {agent_name}\n{safe_finding}\n")

    # --- Debate History ---
    if debate_history:
        sections.append("## Debate History\n")
        for i, entry in enumerate(debate_history, 1):
            safe_entry = redact_pii(entry)
            sections.append(f"**Round {i}:** {safe_entry}\n")

    output = "\n".join(sections)

    # Add disclaimer
    output = add_disclaimer(output)

    return output


# ---------------------------------------------------------------------------
# Streamlit UI (only runs when executed directly, not during import)
# ---------------------------------------------------------------------------


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="MedGemma Council",
        page_icon="🏥",
        layout="wide",
    )

    st.title("MedGemma Council of Experts")
    st.caption(
        "Multi-agent clinical decision support powered by MedGemma 1.5"
    )

    # Initialize session state
    if "council_state" not in st.session_state:
        st.session_state.council_state = create_initial_state()
    if "results" not in st.session_state:
        st.session_state.results = None

    # --- Sidebar: Patient Context Form ---
    with st.sidebar:
        st.header("Patient Information")

        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        chief_complaint = st.text_area(
            "Chief Complaint", placeholder="e.g., Chest pain radiating to left arm"
        )
        history = st.text_area(
            "Past Medical History",
            placeholder="e.g., Hypertension, Type 2 Diabetes",
        )
        medications = st.text_input(
            "Current Medications (comma-separated)",
            placeholder="e.g., Aspirin, Lisinopril, Metformin",
        )

        st.header("Medical Images")
        uploaded_files = st.file_uploader(
            "Upload X-rays, CT scans, MRI",
            type=["png", "jpg", "jpeg", "dcm"],
            accept_multiple_files=True,
        )

    # --- Main Panel ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Case Submission")

        if st.button("Run Council Analysis", type="primary"):
            if not chief_complaint:
                st.warning("Please enter a chief complaint.")
            else:
                # Build patient context
                patient_ctx = build_patient_context(
                    age=age,
                    sex=sex,
                    chief_complaint=chief_complaint,
                    history=history,
                    medications=medications,
                )

                # Process images
                image_paths = []
                if uploaded_files:
                    image_paths = process_uploaded_images(uploaded_files)

                # Prepare state
                state = create_initial_state()
                state["messages"] = [
                    {"role": "user", "content": chief_complaint}
                ]
                state["patient_context"] = patient_ctx
                state["medical_images"] = image_paths

                # Run council
                with st.spinner("Council deliberating..."):
                    result = run_council(state)

                st.session_state.council_state = result
                st.session_state.results = result

        # Display results
        if st.session_state.results:
            result = st.session_state.results
            output = format_council_output(
                final_plan=result.get("final_plan", ""),
                agent_outputs=result.get("agent_outputs", {}),
                debate_history=result.get("debate_history", []),
            )
            st.markdown(output)

    with col2:
        st.subheader("Council Status")
        if st.session_state.results:
            result = st.session_state.results
            st.metric("Consensus", "Yes" if result.get("consensus_reached") else "No")
            st.metric("Debate Rounds", result.get("iteration_count", 0))
            st.metric("Specialists", len(result.get("agent_outputs", {})))

            if result.get("conflict_detected"):
                st.warning("Conflict detected between specialists")
            else:
                st.success("No conflicts detected")
        else:
            st.info("Submit a case to start the council analysis.")


if __name__ == "__main__":
    main()
