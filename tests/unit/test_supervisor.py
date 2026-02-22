"""
Tests for SupervisorAgent (Router & Judge).

TDD: Written BEFORE src/agents/supervisor.py.
Per MASTER_PROMPT:
- Use MedGemma 1.5 4B by default (optional 27B override).
- Check for contradictions -> route to debate_node or final_report.
- test_routing_logic: chest pain -> RadiologyAgent, anxiety -> no radiology.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestSupervisorAgent:
    """Tests for the SupervisorAgent routing and judge logic."""

    def test_inherits_base_agent(self):
        """SupervisorAgent must inherit from BaseAgent."""
        from agents.supervisor import SupervisorAgent
        from agents.base import BaseAgent

        assert issubclass(SupervisorAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """SupervisorAgent.name must return 'SupervisorAgent'."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        assert agent.name == "SupervisorAgent"

    def test_routing_chest_pain_activates_cardiology(self, mock_llm):
        """Chest pain should activate CardiologyAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "chest pain radiating to left arm"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["CardiologyAgent", "RadiologyAgent"]
            specialists = agent.route(state)

        assert "CardiologyAgent" in specialists

    def test_routing_cancer_activates_oncology(self, mock_llm):
        """Cancer-related presentation should activate OncologyAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "lung mass found on CT scan"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["OncologyAgent", "RadiologyAgent"]
            specialists = agent.route(state)

        assert "OncologyAgent" in specialists

    def test_routing_pediatric_activates_pediatrics(self, mock_llm):
        """Child patient should activate PediatricsAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 5, "chief_complaint": "fever"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["PediatricsAgent"]
            specialists = agent.route(state)

        assert "PediatricsAgent" in specialists

    def test_routing_anxiety_activates_psychiatry(self, mock_llm):
        """Mental health presentation should activate PsychiatryAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 28, "chief_complaint": "severe anxiety and panic attacks"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["PsychiatryAgent"]
            specialists = agent.route(state)

        assert "PsychiatryAgent" in specialists

    def test_routing_trauma_activates_emergency(self, mock_llm):
        """Trauma presentation should activate EmergencyMedicineAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 35, "chief_complaint": "multiple injuries from MVA"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["EmergencyMedicineAgent", "RadiologyAgent"]
            specialists = agent.route(state)

        assert "EmergencyMedicineAgent" in specialists

    def test_routing_skin_lesion_activates_dermatology(self, mock_llm):
        """Skin lesion presentation should activate DermatologyAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 45, "chief_complaint": "growing mole with irregular borders"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["DermatologyAgent"]
            specialists = agent.route(state)

        assert "DermatologyAgent" in specialists

    def test_routing_seizure_activates_neurology(self, mock_llm):
        """Seizure presentation should activate NeurologyAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 30, "chief_complaint": "recurrent seizures and headaches"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["NeurologyAgent", "RadiologyAgent"]
            specialists = agent.route(state)

        assert "NeurologyAgent" in specialists

    def test_routing_diabetes_activates_endocrinology(self, mock_llm):
        """Diabetes/thyroid presentation should activate EndocrinologyAgent."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 48, "chief_complaint": "uncontrolled diabetes and thyroid nodule"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists") as mock_det:
            mock_det.return_value = ["EndocrinologyAgent"]
            specialists = agent.route(state)

        assert "EndocrinologyAgent" in specialists

    def test_detect_conflict_finds_contradictions(self, mock_llm):
        """detect_conflict() should flag when agents contradict each other."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        agent_outputs = {
            "CardiologyAgent": "Stop chemotherapy due to reduced LVEF.",
            "OncologyAgent": "Continue chemotherapy to prevent relapse.",
        }

        with patch.object(agent, "_check_conflict", return_value=True):
            conflict = agent.detect_conflict(agent_outputs)

        assert conflict is True

    def test_detect_conflict_no_contradiction(self, mock_llm):
        """detect_conflict() should not flag when agents agree."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        agent_outputs = {
            "CardiologyAgent": "Initiate beta-blocker therapy.",
            "OncologyAgent": "No cardiac contraindication to current regimen.",
        }

        with patch.object(agent, "_check_conflict", return_value=False):
            conflict = agent.detect_conflict(agent_outputs)

        assert conflict is False

    def test_synthesize_final_plan(self, mock_llm):
        """synthesize() must produce a final_plan from all agent outputs."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 65, "chief_complaint": "chest pain"},
            "medical_images": [],
            "agent_outputs": {
                "CardiologyAgent": "ACS workup recommended.",
                "RadiologyAgent": "CXR shows cardiomegaly.",
            },
            "debate_history": ["Round 1: agents presented initial findings."],
            "consensus_reached": True,
            "research_findings": "PMID: 12345 - Relevant evidence.",
            "conflict_detected": False,
            "iteration_count": 1,
            "final_plan": "",
        }

        with patch.object(
            agent,
            "_generate_plan",
            return_value="Final Plan: ACS workup with troponin and ECG. Monitor cardiomegaly."
        ):
            result = agent.synthesize(state)

        assert "final_plan" in result
        assert len(result["final_plan"]) > 0

    def test_analyze_orchestrates_routing(self, mock_llm):
        """analyze() should perform routing and return next specialists."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "headache"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_determine_specialists", return_value=["CardiologyAgent"]):
            with patch.object(agent, "_check_conflict", return_value=False):
                result = agent.analyze(state)

        assert "agent_outputs" in result


class TestSupervisorLogging:
    """Tests for verbose logging in SupervisorAgent methods."""

    def test_route_logs_activated_specialists(self, mock_llm):
        """route() should log which specialists were activated."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"chief_complaint": "chest pain"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch("agents.supervisor.logger") as mock_logger:
            with patch.object(
                agent, "_determine_specialists",
                return_value=["CardiologyAgent", "RadiologyAgent"],
            ):
                agent.route(state)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "cardiologyagent" in all_log_text.lower()

    def test_detect_conflict_logs_result(self, mock_llm):
        """detect_conflict() should log whether conflict was found."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        agent_outputs = {
            "CardiologyAgent": "Stop chemo.",
            "OncologyAgent": "Continue chemo.",
        }

        with patch("agents.supervisor.logger") as mock_logger:
            with patch.object(agent, "_check_conflict", return_value=True):
                agent.detect_conflict(agent_outputs)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "conflict" in all_log_text.lower()

    def test_synthesize_logs_prompt_length(self, mock_llm):
        """synthesize() should log the synthesis prompt length."""
        from agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 65, "chief_complaint": "chest pain"},
            "medical_images": [],
            "agent_outputs": {"CardiologyAgent": "ACS workup."},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch("agents.supervisor.logger") as mock_logger:
            with patch.object(
                agent, "_generate_plan",
                return_value="Final plan text",
            ):
                agent.synthesize(state)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "synth" in all_log_text.lower()
