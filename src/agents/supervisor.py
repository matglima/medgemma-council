"""
SupervisorAgent: Router & Judge for the MedGemma-Council.

Uses MedGemma-27B (Quantized) to:
1. Route cases to appropriate specialist agents.
2. Detect conflicts between specialist outputs.
3. Trigger debate/research cycles when needed.
4. Synthesize the final clinical plan.

Per MASTER_PROMPT:
- If conflict -> Route to debate_node.
- If consensus or max_turns -> Route to final_report.
"""

import logging
from typing import Any, Dict, List

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Orchestrator agent that manages the council workflow.

    Responsibilities:
    - Case ingestion and specialist routing
    - Conflict detection between specialist outputs
    - Debate cycle management
    - Final plan synthesis (Judge role)
    """

    def __init__(self, llm: Any, system_prompt: str = "") -> None:
        default_prompt = (
            "You are a senior clinical supervisor overseeing a multidisciplinary "
            "medical council. Your role is to: "
            "1) Analyze the patient case and identify which specialists to consult, "
            "2) Detect contradictions between specialist recommendations, "
            "3) Facilitate evidence-based debate to resolve conflicts, "
            "4) Synthesize a final, actionable clinical management plan. "
            "Always ensure safety by checking for red flags and missed contraindications."
        )
        super().__init__(llm=llm, system_prompt=system_prompt or default_prompt)

    @property
    def name(self) -> str:
        return "SupervisorAgent"

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration step: route specialists and check for conflicts.

        Returns an update to agent_outputs with routing/supervision decisions.
        """
        agent_outputs = state.get("agent_outputs", {})

        # If no specialist outputs yet, perform initial routing
        if not agent_outputs:
            specialists = self.route(state)
            return {
                "agent_outputs": {
                    self.name: f"Routing to specialists: {', '.join(specialists)}"
                }
            }

        # If specialists have reported, check for conflicts
        conflict = self.detect_conflict(agent_outputs)
        return {
            "agent_outputs": {
                self.name: f"Conflict detected: {conflict}. "
                f"Reviewed {len(agent_outputs)} specialist outputs."
            }
        }

    def route(self, state: Dict[str, Any]) -> List[str]:
        """
        Determine which specialist agents to activate for this case.

        Uses the LLM to analyze patient context and select relevant specialists.
        """
        return self._determine_specialists(state)

    def detect_conflict(self, agent_outputs: Dict[str, str]) -> bool:
        """
        Check if specialist outputs contain contradictions.

        Args:
            agent_outputs: Dict mapping agent names to their findings.

        Returns:
            True if a conflict is detected, False otherwise.
        """
        return self._check_conflict(agent_outputs)

    def synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce the final clinical management plan.

        Aggregates all specialist outputs, research findings, and debate
        history into a coherent plan.

        Returns:
            State update with final_plan and consensus_reached=True.
        """
        plan = self._generate_plan(state)
        return {
            "final_plan": plan,
            "consensus_reached": True,
        }

    def _determine_specialists(self, state: Dict[str, Any]) -> List[str]:
        """
        Internal: Use the LLM to determine which specialists to consult.
        Isolated for mocking in tests.

        In production, the LLM analyzes patient_context to select from:
        CardiologyAgent, OncologyAgent, PediatricsAgent, RadiologyAgent.
        """
        raise NotImplementedError("Requires MedGemma-27B model")

    def _check_conflict(self, agent_outputs: Dict[str, str]) -> bool:
        """
        Internal: Use the LLM to detect contradictions between specialists.
        Isolated for mocking in tests.

        In production, the LLM compares agent outputs and identifies
        conflicting recommendations (e.g., "Stop Drug A" vs "Increase Drug A").
        """
        raise NotImplementedError("Requires MedGemma-27B model")

    def _generate_plan(self, state: Dict[str, Any]) -> str:
        """
        Internal: Use the LLM to synthesize the final clinical plan.
        Isolated for mocking in tests.

        In production, aggregates all state information into a prompt
        and generates a comprehensive management plan.
        """
        raise NotImplementedError("Requires MedGemma-27B model")
