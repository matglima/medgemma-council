"""
Tests for Specialist Agents (Cardiology, Oncology, Pediatrics,
Psychiatry, EmergencyMedicine, Dermatology, Neurology, Endocrinology).

TDD: Written BEFORE src/agents/specialists.py.
Per CLAUDE.md: Clinical Agents must cite specific guidelines.
Per MASTER_PROMPT: RAG-enabled, guideline-grounded agents.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestCardiologyAgent:
    """Tests for the CardiologyAgent specialist."""

    def test_inherits_base_agent(self):
        """CardiologyAgent must inherit from BaseAgent."""
        from agents.specialists import CardiologyAgent
        from agents.base import BaseAgent

        assert issubclass(CardiologyAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """CardiologyAgent.name must return 'CardiologyAgent'."""
        from agents.specialists import CardiologyAgent

        agent = CardiologyAgent(llm=mock_llm)
        assert agent.name == "CardiologyAgent"

    def test_analyze_returns_output(self, mock_llm, sample_patient_context):
        """analyze() must return agent_outputs with cardiology findings."""
        from agents.specialists import CardiologyAgent

        agent = CardiologyAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": sample_patient_context,
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="Recommend ECG. (ACC/AHA 2023 Guidelines)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "CardiologyAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """CardiologyAgent should have a cardiology-specific system prompt."""
        from agents.specialists import CardiologyAgent

        agent = CardiologyAgent(llm=mock_llm)
        assert "cardiolog" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """CardiologyAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import CardiologyAgent

        agent = CardiologyAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")


class TestOncologyAgent:
    """Tests for the OncologyAgent specialist."""

    def test_inherits_base_agent(self):
        """OncologyAgent must inherit from BaseAgent."""
        from agents.specialists import OncologyAgent
        from agents.base import BaseAgent

        assert issubclass(OncologyAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """OncologyAgent.name must return 'OncologyAgent'."""
        from agents.specialists import OncologyAgent

        agent = OncologyAgent(llm=mock_llm)
        assert agent.name == "OncologyAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with oncology findings."""
        from agents.specialists import OncologyAgent

        agent = OncologyAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 58, "chief_complaint": "lung mass"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="Stage IIIA NSCLC. (NCCN Guidelines v1.2025)"):
            result = agent.analyze(state)

        assert "OncologyAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """OncologyAgent should have an oncology-specific system prompt."""
        from agents.specialists import OncologyAgent

        agent = OncologyAgent(llm=mock_llm)
        assert "oncolog" in agent.system_prompt.lower()


class TestPediatricsAgent:
    """Tests for the PediatricsAgent specialist."""

    def test_inherits_base_agent(self):
        """PediatricsAgent must inherit from BaseAgent."""
        from agents.specialists import PediatricsAgent
        from agents.base import BaseAgent

        assert issubclass(PediatricsAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """PediatricsAgent.name must return 'PediatricsAgent'."""
        from agents.specialists import PediatricsAgent

        agent = PediatricsAgent(llm=mock_llm)
        assert agent.name == "PediatricsAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with pediatric findings."""
        from agents.specialists import PediatricsAgent

        agent = PediatricsAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 5, "chief_complaint": "fever and cough"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="Pediatric pneumonia. (AAP Guidelines)"):
            result = agent.analyze(state)

        assert "PediatricsAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """PediatricsAgent should have a pediatrics-specific system prompt."""
        from agents.specialists import PediatricsAgent

        agent = PediatricsAgent(llm=mock_llm)
        assert "pediatric" in agent.system_prompt.lower()

    def test_age_weight_check_flag(self, mock_llm):
        """PediatricsAgent must have weight-based dosing awareness."""
        from agents.specialists import PediatricsAgent

        agent = PediatricsAgent(llm=mock_llm)
        assert hasattr(agent, "enforce_weight_check")
        assert agent.enforce_weight_check is True


class TestPsychiatryAgent:
    """Tests for the PsychiatryAgent specialist."""

    def test_inherits_base_agent(self):
        """PsychiatryAgent must inherit from BaseAgent."""
        from agents.specialists import PsychiatryAgent
        from agents.base import BaseAgent

        assert issubclass(PsychiatryAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """PsychiatryAgent.name must return 'PsychiatryAgent'."""
        from agents.specialists import PsychiatryAgent

        agent = PsychiatryAgent(llm=mock_llm)
        assert agent.name == "PsychiatryAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with psychiatry findings."""
        from agents.specialists import PsychiatryAgent

        agent = PsychiatryAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 34, "chief_complaint": "persistent anxiety and insomnia"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="GAD diagnosis. (APA Practice Guidelines, DSM-5-TR)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "PsychiatryAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """PsychiatryAgent should have a psychiatry-specific system prompt."""
        from agents.specialists import PsychiatryAgent

        agent = PsychiatryAgent(llm=mock_llm)
        assert "psychiatr" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """PsychiatryAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import PsychiatryAgent

        agent = PsychiatryAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")

    def test_has_suicide_risk_awareness(self, mock_llm):
        """PsychiatryAgent must have suicide risk assessment flag."""
        from agents.specialists import PsychiatryAgent

        agent = PsychiatryAgent(llm=mock_llm)
        assert hasattr(agent, "enforce_suicide_screening")
        assert agent.enforce_suicide_screening is True


class TestEmergencyMedicineAgent:
    """Tests for the EmergencyMedicineAgent specialist."""

    def test_inherits_base_agent(self):
        """EmergencyMedicineAgent must inherit from BaseAgent."""
        from agents.specialists import EmergencyMedicineAgent
        from agents.base import BaseAgent

        assert issubclass(EmergencyMedicineAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """EmergencyMedicineAgent.name must return 'EmergencyMedicineAgent'."""
        from agents.specialists import EmergencyMedicineAgent

        agent = EmergencyMedicineAgent(llm=mock_llm)
        assert agent.name == "EmergencyMedicineAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with EM findings."""
        from agents.specialists import EmergencyMedicineAgent

        agent = EmergencyMedicineAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 42, "chief_complaint": "severe abdominal pain with fever"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="ESI Level 2. Acute abdomen workup. (ACLS/ATLS Guidelines)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "EmergencyMedicineAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """EmergencyMedicineAgent should have an EM-specific system prompt."""
        from agents.specialists import EmergencyMedicineAgent

        agent = EmergencyMedicineAgent(llm=mock_llm)
        assert "emergency" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """EmergencyMedicineAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import EmergencyMedicineAgent

        agent = EmergencyMedicineAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")

    def test_has_triage_awareness(self, mock_llm):
        """EmergencyMedicineAgent must have triage protocol flag."""
        from agents.specialists import EmergencyMedicineAgent

        agent = EmergencyMedicineAgent(llm=mock_llm)
        assert hasattr(agent, "enforce_triage_protocol")
        assert agent.enforce_triage_protocol is True


class TestDermatologyAgent:
    """Tests for the DermatologyAgent specialist."""

    def test_inherits_base_agent(self):
        """DermatologyAgent must inherit from BaseAgent."""
        from agents.specialists import DermatologyAgent
        from agents.base import BaseAgent

        assert issubclass(DermatologyAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """DermatologyAgent.name must return 'DermatologyAgent'."""
        from agents.specialists import DermatologyAgent

        agent = DermatologyAgent(llm=mock_llm)
        assert agent.name == "DermatologyAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with dermatology findings."""
        from agents.specialists import DermatologyAgent

        agent = DermatologyAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 55, "chief_complaint": "irregular mole with color changes"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="ABCDE criteria met. Excisional biopsy. (AAD Guidelines 2024)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "DermatologyAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """DermatologyAgent should have a dermatology-specific system prompt."""
        from agents.specialists import DermatologyAgent

        agent = DermatologyAgent(llm=mock_llm)
        assert "dermatolog" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """DermatologyAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import DermatologyAgent

        agent = DermatologyAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")

    def test_has_vision_support_flag(self, mock_llm):
        """DermatologyAgent must have vision/dermoscopy support flag."""
        from agents.specialists import DermatologyAgent

        agent = DermatologyAgent(llm=mock_llm)
        assert hasattr(agent, "supports_dermoscopy")
        assert agent.supports_dermoscopy is True


class TestNeurologyAgent:
    """Tests for the NeurologyAgent specialist."""

    def test_inherits_base_agent(self):
        """NeurologyAgent must inherit from BaseAgent."""
        from agents.specialists import NeurologyAgent
        from agents.base import BaseAgent

        assert issubclass(NeurologyAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """NeurologyAgent.name must return 'NeurologyAgent'."""
        from agents.specialists import NeurologyAgent

        agent = NeurologyAgent(llm=mock_llm)
        assert agent.name == "NeurologyAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with neurology findings."""
        from agents.specialists import NeurologyAgent

        agent = NeurologyAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 62, "chief_complaint": "acute onset left-sided weakness"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="Acute ischemic stroke suspected. tPA within window. (AHA/ASA Stroke Guidelines 2019)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "NeurologyAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """NeurologyAgent should have a neurology-specific system prompt."""
        from agents.specialists import NeurologyAgent

        agent = NeurologyAgent(llm=mock_llm)
        assert "neurolog" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """NeurologyAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import NeurologyAgent

        agent = NeurologyAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")

    def test_has_stroke_awareness_flag(self, mock_llm):
        """NeurologyAgent must have stroke pathway awareness flag."""
        from agents.specialists import NeurologyAgent

        agent = NeurologyAgent(llm=mock_llm)
        assert hasattr(agent, "enforce_stroke_protocol")
        assert agent.enforce_stroke_protocol is True


class TestEndocrinologyAgent:
    """Tests for the EndocrinologyAgent specialist."""

    def test_inherits_base_agent(self):
        """EndocrinologyAgent must inherit from BaseAgent."""
        from agents.specialists import EndocrinologyAgent
        from agents.base import BaseAgent

        assert issubclass(EndocrinologyAgent, BaseAgent)

    def test_name_property(self, mock_llm):
        """EndocrinologyAgent.name must return 'EndocrinologyAgent'."""
        from agents.specialists import EndocrinologyAgent

        agent = EndocrinologyAgent(llm=mock_llm)
        assert agent.name == "EndocrinologyAgent"

    def test_analyze_returns_output(self, mock_llm):
        """analyze() must return agent_outputs with endocrinology findings."""
        from agents.specialists import EndocrinologyAgent

        agent = EndocrinologyAgent(llm=mock_llm)
        state = {
            "messages": [],
            "patient_context": {"age": 52, "chief_complaint": "uncontrolled type 2 diabetes with HbA1c 10.2%"},
            "medical_images": [],
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_run_inference", return_value="Intensify insulin regimen. Consider GLP-1 RA. (ADA Standards of Care 2025)"):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "EndocrinologyAgent" in result["agent_outputs"]

    def test_has_system_prompt_with_persona(self, mock_llm):
        """EndocrinologyAgent should have an endocrinology-specific system prompt."""
        from agents.specialists import EndocrinologyAgent

        agent = EndocrinologyAgent(llm=mock_llm)
        assert "endocrinolog" in agent.system_prompt.lower()

    def test_uses_rag_tool(self, mock_llm):
        """EndocrinologyAgent should have a RAG tool for guideline retrieval."""
        from agents.specialists import EndocrinologyAgent

        agent = EndocrinologyAgent(llm=mock_llm)
        assert hasattr(agent, "rag_tool")

    def test_has_diabetes_management_flag(self, mock_llm):
        """EndocrinologyAgent must have diabetes management protocol flag."""
        from agents.specialists import EndocrinologyAgent

        agent = EndocrinologyAgent(llm=mock_llm)
        assert hasattr(agent, "enforce_diabetes_protocol")
        assert agent.enforce_diabetes_protocol is True
