"""
Tests for RadiologyAgent (MedGemma 1.5 4B Vision).

TDD: Written BEFORE src/agents/radiology.py.
Per MASTER_PROMPT: Must support volumetric/longitudinal input.
Prompt logic: "Compare Image A (Current) with Image B (Prior)."
"""

import pytest
from unittest.mock import MagicMock, patch


class TestRadiologyAgent:
    """Tests for the RadiologyAgent vision agent."""

    def test_inherits_base_agent(self):
        """RadiologyAgent must inherit from BaseAgent."""
        from agents.radiology import RadiologyAgent
        from agents.base import BaseAgent

        assert issubclass(RadiologyAgent, BaseAgent)

    def test_name_property(self, mock_vision_model):
        """RadiologyAgent.name must return 'RadiologyAgent'."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        assert agent.name == "RadiologyAgent"

    def test_analyze_returns_agent_output(self, mock_vision_model, sample_medical_images):
        """analyze() must return a dict with agent_outputs containing radiology findings."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        state = {
            "messages": [],
            "patient_context": {"age": 55, "chief_complaint": "cough"},
            "medical_images": sample_medical_images,
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_process_images", return_value="Opacity in RLL consistent with pneumonia."):
            result = agent.analyze(state)

        assert "agent_outputs" in result
        assert "RadiologyAgent" in result["agent_outputs"]
        assert "pneumonia" in result["agent_outputs"]["RadiologyAgent"].lower()

    def test_analyze_image_sequence(self, mock_vision_model, sample_medical_images):
        """Must handle a sequence of images (volumetric/longitudinal)."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        state = {
            "messages": [],
            "patient_context": {"age": 60},
            "medical_images": sample_medical_images,  # 3 images
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(
            agent,
            "_process_images",
            return_value="Volumetric analysis: 3 slices processed. Tumor size 2.1cm in RUL."
        ):
            result = agent.analyze(state)

        output = result["agent_outputs"]["RadiologyAgent"]
        assert "slices" in output.lower() or "volumetric" in output.lower()

    def test_analyze_no_images_returns_message(self, mock_vision_model):
        """analyze() with no images should return an informative message."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        state = {
            "messages": [],
            "patient_context": {"age": 40},
            "medical_images": [],  # No images
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        result = agent.analyze(state)
        output = result["agent_outputs"]["RadiologyAgent"]
        assert "no" in output.lower() and "image" in output.lower()

    def test_process_images_called_with_paths(self, mock_vision_model, sample_medical_images):
        """_process_images must receive the image paths from state."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        state = {
            "messages": [],
            "patient_context": {},
            "medical_images": sample_medical_images,
            "agent_outputs": {},
            "debate_history": [],
            "consensus_reached": False,
            "research_findings": "",
            "conflict_detected": False,
            "iteration_count": 0,
            "final_plan": "",
        }

        with patch.object(agent, "_process_images", return_value="findings") as mock_proc:
            agent.analyze(state)
            mock_proc.assert_called_once_with(sample_medical_images)

    def test_longitudinal_comparison_prompt(self, mock_vision_model, sample_medical_images):
        """When multiple images exist, agent should frame as longitudinal comparison."""
        from agents.radiology import RadiologyAgent

        agent = RadiologyAgent(llm=mock_vision_model)
        # Verify the agent has a method or logic for building comparison prompts
        assert hasattr(agent, "_build_prompt")
        prompt = agent._build_prompt(sample_medical_images, {"age": 55})
        assert "compare" in prompt.lower() or "interval" in prompt.lower() or "prior" in prompt.lower()
