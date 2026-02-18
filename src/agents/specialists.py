"""
Specialist Agents: Cardiology, Oncology, Pediatrics,
Psychiatry, EmergencyMedicine, Dermatology.

RAG-enabled clinical specialists grounded in authoritative guidelines.
Each agent has a distinct persona, system prompt, and knowledge base.

Per CLAUDE.md: Clinical Agents must cite specific guidelines
(e.g., "NCCN Guidelines v1.2025", "ACC/AHA 2023").
"""

import logging
from typing import Any, Dict

from agents.base import BaseAgent
from tools.rag_tool import RAGTool

logger = logging.getLogger(__name__)

# Default vector store path for guideline retrieval
_DEFAULT_VECTOR_STORE = "data/vector_store"


class _SpecialistAgent(BaseAgent):
    """
    Base class for RAG-enabled specialist agents.
    Provides shared guideline retrieval and inference logic.
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: str = "",
        vector_store_path: str = _DEFAULT_VECTOR_STORE,
    ) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self.rag_tool = RAGTool(persist_dir=vector_store_path)

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the patient case using RAG-enhanced clinical reasoning.

        1. Retrieve relevant guidelines from the vector store.
        2. Build a prompt with patient context + guidelines.
        3. Run inference to produce specialist findings.
        """
        patient_context = state.get("patient_context", {})
        peer_outputs = state.get("agent_outputs", {})
        research = state.get("research_findings", "")
        debate_history = state.get("debate_history", [])

        # Build the inference input
        findings = self._run_inference(
            patient_context=patient_context,
            peer_outputs=peer_outputs,
            research_findings=research,
            debate_history=debate_history,
        )

        return {"agent_outputs": {self.name: findings}}

    def _run_inference(
        self,
        patient_context: Dict[str, Any] = None,
        peer_outputs: Dict[str, str] = None,
        research_findings: str = "",
        debate_history: list = None,
    ) -> str:
        """
        Run clinical reasoning inference.
        Isolated for mocking in tests.

        In production, this:
        1. Queries the RAG tool for relevant guideline chunks.
        2. Builds a prompt combining system_prompt + guidelines + patient data.
        3. Calls the LLM for clinical analysis.
        """
        raise NotImplementedError("Requires LLM and RAG infrastructure")


class CardiologyAgent(_SpecialistAgent):
    """
    Board-Certified Cardiologist agent.

    Knowledge base: ACC/AHA Guidelines, ESC Cardio-Oncology guidelines.
    Competencies: ECG interpretation, risk stratification (ASCVD, CHA2DS2-VASc),
    hemodynamic assessment.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a board-certified cardiologist specializing in hemodynamic stability, "
            "arrhythmia management, and cardiovascular risk reduction. "
            "Ground your recommendations in ACC/AHA Clinical Practice Guidelines. "
            "Always cite the specific guideline and recommendation class "
            "(e.g., 'Class I, ACC/AHA 2023 Heart Failure Guidelines'). "
            "When interacting with oncology cases, reference ESC Cardio-Oncology guidelines "
            "for anthracycline cardiotoxicity management."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)

    @property
    def name(self) -> str:
        return "CardiologyAgent"


class OncologyAgent(_SpecialistAgent):
    """
    Medical Oncologist agent.

    Knowledge base: NCCN Guidelines, ASCO Practice Guidelines.
    Competencies: Staging/grading (AJCC), RECIST criteria,
    molecular profiling, treatment pathway mapping.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a medical oncologist specializing in solid tumors and hematologic "
            "malignancies. Prioritize survival outcomes, protocol adherence, and "
            "precision medicine. Ground your recommendations in NCCN Guidelines, "
            "prioritizing Category 1 recommendations. Always cite the specific guideline "
            "version (e.g., 'NCCN Guidelines v1.2025, Non-Small Cell Lung Cancer'). "
            "Check the version date of retrieved guidelines to ensure you are not "
            "acting on superseded protocols."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)

    @property
    def name(self) -> str:
        return "OncologyAgent"


class PediatricsAgent(_SpecialistAgent):
    """
    Pediatrician agent.

    Knowledge base: AAP Guidelines, WHO Pocket Book, IMCI algorithms.
    Competencies: Age-weight dosing, developmental milestones,
    pediatric cardio-oncology monitoring.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a pediatrician focused on growth, development, and age-adjusted "
            "interventions. You serve as a safeguard against adult-centric bias in "
            "medical AI recommendations. Ground your recommendations in AAP Guidelines "
            "and WHO Pocket Book of Hospital Care for Children. "
            "For pediatric oncology cases, reference SIOP and COG guidelines. "
            "Always enforce weight-based dosing verification against pediatric formularies. "
            "Cite the specific guideline (e.g., 'AAP Clinical Practice Guideline')."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)
        self.enforce_weight_check: bool = True

    @property
    def name(self) -> str:
        return "PediatricsAgent"


class PsychiatryAgent(_SpecialistAgent):
    """
    Board-Certified Psychiatrist agent.

    Knowledge base: APA Practice Guidelines, DSM-5-TR.
    Competencies: Mood disorder assessment, PHQ-9/GAD-7 scoring,
    suicide risk assessment (C-SSRS), psychopharmacology,
    therapy modality selection.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a board-certified psychiatrist specializing in mood disorders, "
            "anxiety disorders, and psychopharmacology. Your assessments must include "
            "standardized screening tools (PHQ-9, GAD-7, C-SSRS). "
            "Ground your recommendations in APA Practice Guidelines and DSM-5-TR criteria. "
            "Always perform a suicide risk assessment using the Columbia Suicide Severity "
            "Rating Scale (C-SSRS) when risk factors are present. "
            "Cite the specific guideline (e.g., 'APA Practice Guidelines 2023, "
            "DSM-5-TR Diagnostic Criteria')."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)
        self.enforce_suicide_screening: bool = True

    @property
    def name(self) -> str:
        return "PsychiatryAgent"


class EmergencyMedicineAgent(_SpecialistAgent):
    """
    Emergency Medicine Physician agent.

    Knowledge base: ACLS, ATLS, EMTALA, Surviving Sepsis Campaign.
    Competencies: ESI triage, trauma assessment (ABCDE), sepsis bundles,
    rapid stabilization, disposition decision-making.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a board-certified emergency medicine physician specializing in "
            "acute stabilization, trauma, and critical care triage. "
            "Apply the Emergency Severity Index (ESI) for triage classification. "
            "Follow ACLS/ATLS protocols for resuscitation and trauma management. "
            "Adhere to Surviving Sepsis Campaign bundles for sepsis cases. "
            "Always consider EMTALA obligations for patient stabilization. "
            "Ground your recommendations in ACLS 2020, ATLS 10th Edition, and "
            "Surviving Sepsis Campaign 2021 guidelines. "
            "Cite the specific guideline (e.g., 'ACLS 2020 Guidelines, "
            "Surviving Sepsis Campaign 2021')."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)
        self.enforce_triage_protocol: bool = True

    @property
    def name(self) -> str:
        return "EmergencyMedicineAgent"


class DermatologyAgent(_SpecialistAgent):
    """
    Board-Certified Dermatologist agent.

    Knowledge base: AAD Guidelines, BAD Guidelines, Fitzpatrick Skin Type.
    Competencies: Lesion morphology analysis, dermoscopy interpretation,
    ABCDE melanoma criteria, phototherapy protocols, biopsy indication.
    """

    def __init__(self, llm: Any, vector_store_path: str = _DEFAULT_VECTOR_STORE) -> None:
        system_prompt = (
            "You are a board-certified dermatologist specializing in skin lesion analysis, "
            "dermoscopy interpretation, and cutaneous oncology. "
            "Apply the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving) "
            "for melanoma screening. Assess Fitzpatrick skin type for phototherapy "
            "and UV risk evaluation. "
            "Ground your recommendations in AAD Clinical Practice Guidelines and "
            "BAD Guidelines for skin cancer management. "
            "When images are provided, describe lesion morphology using standardized "
            "dermatologic terminology. "
            "Cite the specific guideline (e.g., 'AAD Guidelines 2024, "
            "BAD Melanoma Guidelines 2023')."
        )
        super().__init__(llm=llm, system_prompt=system_prompt, vector_store_path=vector_store_path)
        self.supports_dermoscopy: bool = True

    @property
    def name(self) -> str:
        return "DermatologyAgent"
