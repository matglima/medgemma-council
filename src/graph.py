"""
LangGraph state machine definition for MedGemma-Council.

Defines the CouncilState schema — the shared state object that flows
through all nodes in the multi-agent debate graph.
"""

from typing import Any, Dict, List, TypedDict


class CouncilState(TypedDict):
    """
    Shared state dictionary persisted across the LangGraph execution.
    Acts as the 'medical record' of the council conversation.

    Core fields (from spec Section 3, MASTER_PROMPT):
        messages: List of LangChain-style message dicts.
        patient_context: Dict with age, symptoms, history, vitals, labs.
        medical_images: List of file paths (CT slices, X-rays, etc.).
        agent_outputs: Dict mapping 'AgentName' -> latest finding string.
        debate_history: List of strings tracking argument rounds.
        consensus_reached: Boolean flag indicating debate termination.

    Orchestration fields (from RESEARCH_REPORT Section 3.1.1):
        research_findings: Output from the Research Agent (PubMed summaries).
        conflict_detected: Flag indicating disagreement among specialists.
        iteration_count: Counter to prevent infinite debate loops.
        final_plan: The synthesized clinical plan produced by the Judge.
    """

    # Core fields
    messages: List[Dict[str, Any]]
    patient_context: Dict[str, Any]
    medical_images: List[str]
    agent_outputs: Dict[str, str]
    debate_history: List[str]
    consensus_reached: bool

    # Orchestration fields
    research_findings: str
    conflict_detected: bool
    iteration_count: int
    final_plan: str
