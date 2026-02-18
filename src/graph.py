"""
LangGraph state machine definition for MedGemma-Council.

Defines:
- CouncilState: The shared state TypedDict flowing through all nodes.
- Node functions: ingestion, routing, specialist analysis, conflict check,
  research, debate, synthesis.
- build_council_graph(): Constructs and compiles the LangGraph StateGraph.

Graph topology:
  ingestion -> supervisor_route -> specialist -> conflict_check
    -> (conflict + under max iterations) -> research -> debate -> conflict_check
    -> (no conflict OR max iterations) -> synthesis -> END
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.supervisor import SupervisorAgent
from agents.researcher import ResearchAgent

logger = logging.getLogger(__name__)

# Maximum debate rounds before forcing synthesis
MAX_DEBATE_ROUNDS = 3


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Agent factories (create fresh instances; tests patch the classes)
# ---------------------------------------------------------------------------


def _make_supervisor() -> SupervisorAgent:
    """Create a SupervisorAgent. Tests patch graph.SupervisorAgent."""
    from unittest.mock import MagicMock
    return SupervisorAgent(llm=MagicMock())


def _make_researcher() -> ResearchAgent:
    """Create a ResearchAgent. Tests patch graph.ResearchAgent."""
    from unittest.mock import MagicMock
    return ResearchAgent(llm=MagicMock(), pubmed_email="council@medgemma.ai")


# ---------------------------------------------------------------------------
# Internal helpers (isolated for mocking in tests)
# ---------------------------------------------------------------------------


def _run_specialists(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Run activated specialist agents and collect their outputs.
    Isolated for mocking in tests.

    In production, instantiates the specialist agents determined by the
    supervisor and runs them against the current state.
    """
    raise NotImplementedError("Requires specialist agent infrastructure")


def _run_debate_round(state: Dict[str, Any]) -> List[str]:
    """
    Run one round of structured debate between specialists.
    Isolated for mocking in tests.

    In production, each specialist critiques others' outputs using
    the latest research findings as evidence.
    """
    raise NotImplementedError("Requires specialist agent infrastructure")


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------


def ingestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enrich the initial patient state.

    - Ensures required fields are present.
    - Resets orchestration counters for a fresh run.
    """
    logger.info("Ingestion node: validating patient state")

    return {
        "iteration_count": 0,
        "consensus_reached": False,
        "conflict_detected": False,
        "final_plan": "",
    }


def supervisor_route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor analyzes the case and determines which specialists to consult.
    """
    logger.info("Supervisor routing node: determining specialists")

    supervisor = _make_supervisor()
    specialists = supervisor.route(state)

    return {
        "agent_outputs": {
            supervisor.name: f"Routing to specialists: {', '.join(specialists)}"
        },
    }


def specialist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the activated specialist agents and collect findings.
    """
    logger.info("Specialist node: running specialist analysis")

    specialist_outputs = _run_specialists(state)

    # Merge specialist outputs with existing outputs
    current_outputs = dict(state.get("agent_outputs", {}))
    current_outputs.update(specialist_outputs)

    return {"agent_outputs": current_outputs}


def conflict_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor checks for contradictions between specialist outputs.
    """
    logger.info("Conflict check node: detecting disagreements")

    supervisor = _make_supervisor()
    agent_outputs = state.get("agent_outputs", {})
    conflict = supervisor.detect_conflict(agent_outputs)

    return {"conflict_detected": conflict}


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research agent fetches PubMed literature to resolve conflicts.
    """
    logger.info("Research node: fetching literature for conflict resolution")

    researcher = _make_researcher()
    result = researcher.analyze(state)

    return {"research_findings": result.get("research_findings", "")}


def debate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specialists debate using research evidence. Increments iteration count.
    """
    logger.info("Debate node: structured specialist debate")

    new_arguments = _run_debate_round(state)

    current_history = list(state.get("debate_history", []))
    current_history.extend(new_arguments)

    return {
        "debate_history": current_history,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor synthesizes all outputs into a final clinical plan.
    """
    logger.info("Synthesis node: producing final clinical plan")

    supervisor = _make_supervisor()
    result = supervisor.synthesize(state)

    return {
        "final_plan": result.get("final_plan", ""),
        "consensus_reached": result.get("consensus_reached", True),
    }


# ---------------------------------------------------------------------------
# Conditional edge logic
# ---------------------------------------------------------------------------


def _should_continue_after_conflict(state: Dict[str, Any]) -> str:
    """
    After conflict check, decide whether to debate or synthesize.

    Returns:
        "research" if conflict detected and under max iterations.
        "synthesis" if no conflict or max iterations reached.
    """
    conflict = state.get("conflict_detected", False)
    iteration = state.get("iteration_count", 0)

    if conflict and iteration < MAX_DEBATE_ROUNDS:
        logger.info(
            f"Conflict detected (iteration {iteration}/{MAX_DEBATE_ROUNDS}), "
            "routing to research -> debate"
        )
        return "research"

    if conflict:
        logger.warning(
            f"Max debate rounds ({MAX_DEBATE_ROUNDS}) reached, forcing synthesis"
        )
    else:
        logger.info("No conflict detected, proceeding to synthesis")

    return "synthesis"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------


def build_council_graph():
    """
    Construct and compile the MedGemma-Council LangGraph state machine.

    Topology:
        ingestion -> supervisor_route -> specialist -> conflict_check
          -> [conflict & under max] -> research -> debate -> conflict_check
          -> [no conflict | max reached] -> synthesis -> END

    Returns:
        A compiled LangGraph application ready for .invoke().
    """
    graph = StateGraph(CouncilState)

    # Add nodes
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("supervisor_route", supervisor_route_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("conflict_check", conflict_check_node)
    graph.add_node("research", research_node)
    graph.add_node("debate", debate_node)
    graph.add_node("synthesis", synthesis_node)

    # Linear edges
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "supervisor_route")
    graph.add_edge("supervisor_route", "specialist")
    graph.add_edge("specialist", "conflict_check")

    # Conditional: after conflict check, either debate or synthesize
    graph.add_conditional_edges(
        "conflict_check",
        _should_continue_after_conflict,
        {
            "research": "research",
            "synthesis": "synthesis",
        },
    )

    # Debate loop: research -> debate -> back to conflict_check
    graph.add_edge("research", "debate")
    graph.add_edge("debate", "conflict_check")

    # Synthesis terminates the graph
    graph.add_edge("synthesis", END)

    return graph.compile()
