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
        CardiologyAgent, OncologyAgent, PediatricsAgent, RadiologyAgent,
        PsychiatryAgent, EmergencyMedicineAgent, DermatologyAgent,
        NeurologyAgent, EndocrinologyAgent.
        """
        patient_context = state.get("patient_context", {})
        images = state.get("medical_images", [])

        prompt = (
            f"{self.system_prompt}\n\n"
            f"Patient Context:\n"
            f"  Age: {patient_context.get('age', 'unknown')}\n"
            f"  Sex: {patient_context.get('sex', 'unknown')}\n"
            f"  Chief Complaint: {patient_context.get('chief_complaint', 'not specified')}\n"
            f"  History: {patient_context.get('history', 'not provided')}\n"
            f"  Medications: {patient_context.get('medications', [])}\n"
            f"  Images available: {len(images)}\n\n"
            f"Available specialists: CardiologyAgent, OncologyAgent, PediatricsAgent, "
            f"RadiologyAgent, PsychiatryAgent, EmergencyMedicineAgent, DermatologyAgent, "
            f"NeurologyAgent, EndocrinologyAgent.\n\n"
            f"Return ONLY a comma-separated list of specialist names to activate "
            f"(e.g., 'CardiologyAgent, RadiologyAgent'). Do not include explanations."
        )

        result = self.llm(prompt, max_tokens=256)
        if isinstance(result, dict):
            text = result["choices"][0]["text"]
        else:
            text = str(result)

        # Parse the comma-separated list
        all_specialists = [
            "CardiologyAgent", "OncologyAgent", "PediatricsAgent",
            "RadiologyAgent", "PsychiatryAgent", "EmergencyMedicineAgent",
            "DermatologyAgent", "NeurologyAgent", "EndocrinologyAgent",
        ]
        activated = []
        for name in all_specialists:
            if name.lower() in text.lower():
                activated.append(name)

        # Always include RadiologyAgent if images are present
        if images and "RadiologyAgent" not in activated:
            activated.append("RadiologyAgent")

        # Fallback: if nothing matched, use a general specialist
        if not activated:
            activated = ["CardiologyAgent"]

        return activated

    def _check_conflict(self, agent_outputs: Dict[str, str]) -> bool:
        """
        Internal: Use the LLM to detect contradictions between specialists.
        Isolated for mocking in tests.

        In production, the LLM compares agent outputs and identifies
        conflicting recommendations (e.g., "Stop Drug A" vs "Increase Drug A").
        """
        if len(agent_outputs) < 2:
            return False

        # Filter out the supervisor's own output
        specialist_outputs = {
            k: v for k, v in agent_outputs.items()
            if k != self.name
        }

        if len(specialist_outputs) < 2:
            return False

        outputs_text = "\n".join(
            f"  {name}: {finding}" for name, finding in specialist_outputs.items()
        )

        prompt = (
            f"{self.system_prompt}\n\n"
            f"The following specialist agents have provided their findings:\n"
            f"{outputs_text}\n\n"
            f"Are there any direct contradictions between these recommendations? "
            f"Reply with ONLY 'CONFLICT' or 'NO_CONFLICT'."
        )

        result = self.llm(prompt, max_tokens=64)
        if isinstance(result, dict):
            text = result["choices"][0]["text"]
        else:
            text = str(result)

        return "conflict" in text.lower() and "no_conflict" not in text.lower()

    def _generate_plan(self, state: Dict[str, Any]) -> str:
        """
        Internal: Use the LLM to synthesize the final clinical plan.
        Isolated for mocking in tests.

        Aggregates all state information into a prompt and generates
        a comprehensive management plan with citations.
        """
        agent_outputs = state.get("agent_outputs", {})
        research = state.get("research_findings", "")
        debate_history = state.get("debate_history", [])
        patient_context = state.get("patient_context", {})

        prompt_parts = [
            self.system_prompt,
            "",
            f"Patient: Age {patient_context.get('age', 'unknown')}, "
            f"{patient_context.get('sex', 'unknown')}. "
            f"Chief Complaint: {patient_context.get('chief_complaint', 'not specified')}.",
            "",
            "Specialist Findings:",
        ]
        for name, finding in agent_outputs.items():
            prompt_parts.append(f"  {name}: {finding}")

        if research:
            prompt_parts.append(f"\nResearch Evidence:\n{research}")

        if debate_history:
            prompt_parts.append("\nDebate Summary:")
            for entry in debate_history:
                prompt_parts.append(f"  - {entry}")

        prompt_parts.append(
            "\n\nSynthesize all findings into a final clinical management plan. "
            "Include:\n"
            "1. Primary Diagnosis\n"
            "2. Immediate Actions\n"
            "3. Medications (with dosing if applicable)\n"
            "4. Follow-up Plan\n"
            "5. Red Flags to Watch For\n"
            "Cite specific guidelines for each recommendation."
        )

        prompt = "\n".join(prompt_parts)

        result = self.llm(prompt, max_tokens=2048)
        if isinstance(result, dict):
            return result["choices"][0]["text"]
        return str(result)
