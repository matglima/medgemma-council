"""
LangGraph state machine definition for MedGemma-Council.

Defines:
- CouncilState: The shared state TypedDict flowing through all nodes.
- Node functions: ingestion, routing, specialist analysis, safety check,
  conflict check, research, debate, synthesis, emergency synthesis.
- build_council_graph(): Constructs and compiles the LangGraph StateGraph.

Graph topology:
  ingestion -> supervisor_route -> specialist -> safety_check
    -> [red_flag_detected] -> emergency_synthesis -> END
    -> [safe] -> conflict_check
      -> (conflict + under max iterations) -> research -> debate -> conflict_check
      -> (no conflict OR max iterations) -> synthesis -> END
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.supervisor import SupervisorAgent
from agents.researcher import ResearchAgent
from agents.specialists import (
    CardiologyAgent,
    OncologyAgent,
    PediatricsAgent,
    PsychiatryAgent,
    EmergencyMedicineAgent,
    DermatologyAgent,
    NeurologyAgent,
    EndocrinologyAgent,
)
from agents.radiology import RadiologyAgent

from utils.model_factory import ModelFactory
from utils.safety import scan_for_red_flags

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

    Safety fields (Phase 12):
        red_flag_detected: Whether a clinical red flag was found in agent outputs.
        emergency_override: The emergency referral message if red flag detected.
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

    # Safety fields
    red_flag_detected: bool
    emergency_override: str

    # Research control
    force_research: bool


# ---------------------------------------------------------------------------
# Agent factories (create fresh instances; tests patch the classes)
# ---------------------------------------------------------------------------


def _make_supervisor() -> SupervisorAgent:
    """Create a SupervisorAgent. Tests patch graph.SupervisorAgent."""
    factory = ModelFactory()
    return SupervisorAgent(llm=factory.create_text_model())


def _make_researcher() -> ResearchAgent:
    """Create a ResearchAgent. Tests patch graph.ResearchAgent."""
    factory = ModelFactory()
    return ResearchAgent(llm=factory.create_text_model(), pubmed_email="council@medgemma.ai")


# ---------------------------------------------------------------------------
# Internal helpers (isolated for mocking in tests)
# ---------------------------------------------------------------------------


def _run_single_specialist(
    agent_name: str,
    agent_cls: type,
    llm: Any,
    state: Dict[str, Any],
) -> Dict[str, str]:
    """
    Run a single specialist agent and return its outputs.

    Isolated as a top-level function so it can be submitted to
    ThreadPoolExecutor (lambdas and nested functions are fragile
    across pickling boundaries).

    Returns:
        Dict mapping agent name to output string (or error message).
    """
    try:
        agent = agent_cls(llm=llm)
        result = agent.analyze(state)
        agent_out = result.get("agent_outputs", {})
        return agent_out
    except Exception as e:
        logger.error(f"Agent {agent_name} failed: {e}")
        return {agent_name: f"Error: {e}"}


def _run_specialists(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Run activated specialist agents and collect their outputs.
    Isolated for mocking in tests.

    Uses ``concurrent.futures.ThreadPoolExecutor`` so that specialist LLM
    calls can overlap when parallelism is explicitly enabled.

    By default, specialists run **sequentially** (max_workers=1) to prevent
    CUDA OOM from concurrent model.generate() calls on limited-VRAM GPUs
    (e.g. 2xT4 with 14.6 GiB each).

    Configuration:
        COUNCIL_MAX_WORKERS (env var): Override the default thread-pool size.
            Defaults to 1 (sequential). Set to N to run N specialists in parallel.

    Parses the supervisor's routing output to determine which specialists
    to instantiate and run.
    """
    factory = ModelFactory()

    # Parse which specialists were activated from supervisor output
    supervisor_output = ""
    for key, val in state.get("agent_outputs", {}).items():
        if "routing" in str(val).lower() or "specialist" in str(val).lower():
            supervisor_output = str(val)
            break

    # Map names to agent classes
    agent_registry = {
        "CardiologyAgent": CardiologyAgent,
        "OncologyAgent": OncologyAgent,
        "PediatricsAgent": PediatricsAgent,
        "PsychiatryAgent": PsychiatryAgent,
        "EmergencyMedicineAgent": EmergencyMedicineAgent,
        "DermatologyAgent": DermatologyAgent,
        "NeurologyAgent": NeurologyAgent,
        "EndocrinologyAgent": EndocrinologyAgent,
        "RadiologyAgent": RadiologyAgent,
    }

    # Determine which agents to run
    activated = []
    for name in agent_registry:
        if name.lower() in supervisor_output.lower():
            activated.append(name)

    # Fallback: activate all if no routing info
    if not activated:
        activated = ["CardiologyAgent"]

    # If no images are provided, radiology cannot add value and would force
    # loading the heavy vision model unnecessarily.
    medical_images = state.get("medical_images") or []
    if "RadiologyAgent" in activated and not medical_images:
        activated = [name for name in activated if name != "RadiologyAgent"]
        logger.info("Skipping RadiologyAgent: no medical images provided")
        if not activated:
            activated = ["CardiologyAgent"]

    llm = factory.create_text_model()

    # Create vision model only if RadiologyAgent is activated (avoid loading
    # the 4B multimodal model unnecessarily).  RadiologyAgent calls
    # self.llm(images=..., prompt=...) — the VisionModelWrapper interface.
    # Passing the text model causes AutoModelForCausalLM to reject 'images'.
    vision_llm = None
    if "RadiologyAgent" in activated:
        vision_llm = factory.create_vision_model()

    # Determine thread-pool size
    # Default to sequential (max_workers=1) to prevent CUDA OOM from
    # concurrent model.generate() calls on limited-VRAM GPUs (e.g. 2xT4).
    # Use COUNCIL_MAX_WORKERS env var to re-enable parallelism on larger GPUs.
    env_workers = os.environ.get("COUNCIL_MAX_WORKERS")
    max_workers = int(env_workers) if env_workers else 1

    outputs: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each specialist as a parallel task
        future_to_name = {}
        for agent_name in activated:
            agent_cls = agent_registry[agent_name]
            # RadiologyAgent needs the vision model (VisionModelWrapper);
            # all other agents use the text model (TextModelWrapper).
            agent_llm = vision_llm if agent_name == "RadiologyAgent" else llm
            future = executor.submit(
                _run_single_specialist, agent_name, agent_cls, agent_llm, state
            )
            future_to_name[future] = agent_name

        # Collect results as they complete
        for future in as_completed(future_to_name):
            agent_name = future_to_name[future]
            try:
                result = future.result()
                outputs.update(result)
            except Exception as e:
                logger.error(f"Agent {agent_name} future failed: {e}")
                outputs[agent_name] = f"Error: {e}"

    return outputs


def _run_debate_round(state: Dict[str, Any]) -> List[str]:
    """
    Run one round of structured debate between specialists.
    Isolated for mocking in tests.

    Each specialist critiques others' outputs using the latest
    research findings as evidence.
    """
    factory = ModelFactory()

    agent_outputs = state.get("agent_outputs", {})
    research = state.get("research_findings", "")

    # Filter to specialist outputs only (exclude supervisor)
    specialist_outputs = {
        k: v for k, v in agent_outputs.items()
        if k != "SupervisorAgent"
    }

    if len(specialist_outputs) < 2:
        return ["No debate needed — fewer than 2 specialists."]

    llm = factory.create_text_model()

    arguments = []
    specialist_names = list(specialist_outputs.keys())

    for agent_name in specialist_names:
        # Build a debate prompt for this agent
        own_output = specialist_outputs[agent_name]
        peer_outputs = {
            k: v for k, v in specialist_outputs.items()
            if k != agent_name
        }

        peers_text = "\n".join(
            f"  {name}: {finding}" for name, finding in peer_outputs.items()
        )

        prompt = (
            f"You are {agent_name} in a clinical debate.\n"
            f"Your previous finding: {own_output}\n\n"
            f"Other specialists' findings:\n{peers_text}\n\n"
        )

        if research:
            prompt += f"Latest research evidence:\n{research}\n\n"

        prompt += (
            "Critically evaluate the other specialists' recommendations. "
            "Either revise your position based on new evidence, or defend "
            "your recommendation with specific citations. "
            "Be concise (2-3 sentences)."
        )

        try:
            result = llm(prompt, max_tokens=512)
            if isinstance(result, dict):
                text = result["choices"][0]["text"]
            else:
                text = str(result)
            arguments.append(f"{agent_name}: {text}")
        except Exception as e:
            logger.error(f"Debate failed for {agent_name}: {e}")
            arguments.append(f"{agent_name}: Unable to participate in debate — {e}")

    return arguments


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
        "red_flag_detected": False,
        "emergency_override": "",
        "force_research": False,
    }


def supervisor_route_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor analyzes the case and determines which specialists to consult.
    Also checks if research should be triggered before specialist discussions
    based on case complexity indicators.
    """
    logger.info("Supervisor routing node: determining specialists")

    supervisor = _make_supervisor()
    specialists = supervisor.route(state)

    patient_context = state.get("patient_context", {})
    chief_complaint = patient_context.get("chief_complaint", "").lower()
    history = patient_context.get("history", "").lower()

    research_keywords = [
        "controversial", "unclear", "complex", "rare", "novel",
        "conflicting", "experimental", "guideline", "protocol",
        "clinical trial", "off-label", "multi-disciplinary"
    ]
    needs_research = any(kw in chief_complaint or kw in history for kw in research_keywords)
    user_force_research = state.get("force_research", False)
    should_force_research = needs_research or user_force_research

    if should_force_research:
        logger.info("Supervisor routing: case may benefit from research")

    return {
        "agent_outputs": {
            supervisor.name: f"Routing to specialists: {', '.join(specialists)}"
        },
        "force_research": should_force_research,
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


def safety_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan all agent outputs for clinical red flags.

    If red flags are detected, sets red_flag_detected=True and populates
    emergency_override with the emergency referral message. The conditional
    edge after this node will then route to emergency_synthesis instead
    of the normal conflict_check flow.

    Per CLAUDE.md: "Every agent output must be scanned for Red Flags
    (e.g., suicide risk, sepsis shock). If found, immediately override
    with an emergency referral message."
    """
    logger.info("Safety check node: scanning agent outputs for red flags")

    agent_outputs = state.get("agent_outputs", {})

    # Concatenate all agent outputs for scanning
    all_text = " ".join(str(v) for v in agent_outputs.values())

    result = scan_for_red_flags(all_text)

    if result["flagged"]:
        logger.warning(
            f"RED FLAG DETECTED: {result['flags']}. "
            "Short-circuiting to emergency synthesis."
        )
        return {
            "red_flag_detected": True,
            "emergency_override": result["emergency_message"],
        }

    logger.info("Safety check: no red flags detected, proceeding normally")
    return {
        "red_flag_detected": False,
        "emergency_override": "",
    }


def emergency_synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce an emergency clinical plan when red flags are detected.

    Short-circuits the normal debate/synthesis flow. Includes:
    - The emergency override message
    - All specialist findings that were available at the time
    - Immediate action instructions

    This node immediately terminates the graph.
    """
    logger.info("EMERGENCY synthesis node: producing emergency plan")

    emergency_override = state.get("emergency_override", "")
    agent_outputs = state.get("agent_outputs", {})
    patient_context = state.get("patient_context", {})

    plan_parts = [
        emergency_override,
        "",
        "--- Specialist Findings at Time of Emergency Detection ---",
    ]

    for name, finding in agent_outputs.items():
        if name != "SupervisorAgent":
            plan_parts.append(f"  {name}: {finding}")

    plan_parts.extend([
        "",
        "--- Immediate Actions Required ---",
        "1. Activate emergency medical services (call 911 or equivalent).",
        "2. Initiate appropriate emergency protocols (ACLS, ATLS, etc.).",
        "3. Transfer to nearest emergency department if not already there.",
        "4. Ensure continuous monitoring of vital signs.",
        f"5. Clinical context: {patient_context.get('chief_complaint', 'See above')}.",
        "",
        "This AI system has detected a life-threatening condition and has "
        "terminated normal deliberation. Human clinical judgment is required "
        "IMMEDIATELY.",
    ])

    return {
        "final_plan": "\n".join(plan_parts),
        "consensus_reached": True,
    }


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


def _should_continue_after_safety(state: Dict[str, Any]) -> str:
    """
    After safety check, decide whether to proceed normally or emergency-override.

    Returns:
        "emergency_synthesis" if red flag detected.
        "conflict_check" if safe to proceed normally.
    """
    if state.get("red_flag_detected", False):
        logger.warning("Safety override: routing to emergency synthesis")
        return "emergency_synthesis"

    return "conflict_check"


def _should_continue_after_conflict(state: Dict[str, Any]) -> str:
    """
    After conflict check, decide whether to debate or synthesize.

    Returns:
        "research" if conflict detected (or force_research) and under max iterations.
        "synthesis" if no conflict and not forcing research, or max iterations reached.
    """
    conflict = state.get("conflict_detected", False)
    force_research = state.get("force_research", False)
    iteration = state.get("iteration_count", 0)

    if (conflict or force_research) and iteration < MAX_DEBATE_ROUNDS:
        reason = "Force research" if force_research and not conflict else "Conflict detected"
        logger.info(
            f"{reason} (iteration {iteration}/{MAX_DEBATE_ROUNDS}), "
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


def _should_research_before_specialists(state: Dict[str, Any]) -> str:
    """
    After supervisor routing, decide whether to do research before specialist discussions.

    Returns:
        "do_research_first" if force_research is True, otherwise "run_specialists".
    """
    if state.get("force_research", False):
        logger.info("Force research flag set, routing to research before specialists")
        return "do_research_first"
    return "run_specialists"


def build_council_graph():
    """
    Construct and compile the MedGemma-Council LangGraph state machine.

    Topology:
        ingestion -> supervisor_route
          -> [force_research] -> research -> specialist -> safety_check
          -> [no force] -> specialist -> safety_check
          -> [red_flag] -> emergency_synthesis -> END
          -> [safe] -> conflict_check
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
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("emergency_synthesis", emergency_synthesis_node)
    graph.add_node("conflict_check", conflict_check_node)
    graph.add_node("research", research_node)
    graph.add_node("debate", debate_node)
    graph.add_node("synthesis", synthesis_node)

    # Linear edges
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "supervisor_route")

    # Conditional: after supervisor routing, either do research first or go directly to specialists
    graph.add_conditional_edges(
        "supervisor_route",
        _should_research_before_specialists,
        {
            "do_research_first": "research",
            "run_specialists": "specialist",
        },
    )

    # Research flows to debate (not directly to specialist)
    # The flow is: research -> debate -> conflict_check -> (research OR specialist)
    # This prevents double-calling conflict_check

    graph.add_edge("specialist", "safety_check")

    # Conditional: after safety check, either emergency or normal flow
    graph.add_conditional_edges(
        "safety_check",
        _should_continue_after_safety,
        {
            "emergency_synthesis": "emergency_synthesis",
            "conflict_check": "conflict_check",
        },
    )

    # Emergency synthesis terminates the graph
    graph.add_edge("emergency_synthesis", END)

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
