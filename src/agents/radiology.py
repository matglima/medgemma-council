"""
RadiologyAgent: MedGemma 1.5 4B Vision agent for medical image interpretation.

Supports:
- Volumetric analysis (CT/MRI slices)
- Longitudinal comparison (current vs. prior images)
- Anatomical localization with bounding box awareness

Per MASTER_PROMPT: Implement support for volumetric/longitudinal input.
Prompt logic: "Compare Image A (Current) with Image B (Prior)."
"""

import logging
from typing import Any, Dict, List

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RadiologyAgent(BaseAgent):
    """
    Vision-based radiology agent using MedGemma 1.5 4B.

    Processes medical images (X-rays, CT volumes, MRI) and produces
    structured radiology reports with findings and impressions.
    """

    def __init__(self, llm: Any, system_prompt: str = "") -> None:
        default_prompt = (
            "You are a board-certified radiologist specializing in diagnostic imaging. "
            "Analyze the provided medical images and produce a structured report with: "
            "1) Findings (detailed observations), 2) Impression (clinical interpretation), "
            "3) Comparison with prior studies if available. "
            "Always note interval changes between current and prior images."
        )
        super().__init__(llm=llm, system_prompt=system_prompt or default_prompt)

    @property
    def name(self) -> str:
        return "RadiologyAgent"

    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze medical images from the council state.

        If no images are present, returns an informative message.
        If multiple images exist, performs volumetric/longitudinal analysis.
        """
        images = state.get("medical_images", [])

        if not images:
            return {
                "agent_outputs": {
                    self.name: "No medical images provided for radiological analysis."
                }
            }

        findings = self._process_images(images)
        return {"agent_outputs": {self.name: findings}}

    def _build_prompt(self, image_paths: List[str], patient_context: Dict[str, Any]) -> str:
        """
        Build the inference prompt based on image count and patient context.

        For single images: standard radiology report request.
        For multiple images: longitudinal comparison or volumetric analysis.
        """
        age = patient_context.get("age", "unknown")
        complaint = patient_context.get("chief_complaint", "not specified")

        if len(image_paths) == 1:
            return (
                f"{self.system_prompt}\n\n"
                f"Patient: Age {age}. Chief complaint: {complaint}.\n"
                f"Analyze the provided image and generate a structured radiology report."
            )
        else:
            return (
                f"{self.system_prompt}\n\n"
                f"Patient: Age {age}. Chief complaint: {complaint}.\n"
                f"You are provided {len(image_paths)} images. "
                f"Compare Image A (Current/Latest) with prior images. "
                f"Identify interval changes in opacity, tumor size, effusion, "
                f"or cardiac silhouette. Produce a volumetric/longitudinal report."
            )

    def _process_images(self, image_paths: List[str]) -> str:
        """
        Internal: Run the vision model on the provided images.
        Isolated for mocking in tests.

        In production, this uses MedGemma 1.5 4B multimodal pipeline:
            from PIL import Image
            images = [Image.open(p) for p in image_paths]
            result = self.llm(images=images, prompt=self._build_prompt(...))
            return result[0]["generated_text"]
        """
        raise NotImplementedError("Requires MedGemma 1.5 4B model")
