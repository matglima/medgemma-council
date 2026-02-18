"""
ModelFactory: Centralized model creation with feature flag.

Reads MEDGEMMA_USE_REAL_MODELS env var to decide whether to return
real quantized models or MagicMock instances for testing/development.

Usage:
    factory = ModelFactory()
    text_model = factory.create_text_model()  # Returns mock by default
    vision_model = factory.create_vision_model()

    # To use real models (requires GPU + model weights):
    # export MEDGEMMA_USE_REAL_MODELS=true
"""

import logging
import os
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

# Default model IDs
DEFAULT_TEXT_MODEL_ID = "google/medgemma-27b-text-it"
DEFAULT_VISION_MODEL_ID = "google/medgemma-4b-it"


class ModelFactory:
    """
    Factory for creating text and vision models.

    Controlled by MEDGEMMA_USE_REAL_MODELS env var:
    - False (default): Returns MagicMock instances for testing.
    - True: Loads real quantized models via transformers + bitsandbytes.
    """

    def __init__(self) -> None:
        env_val = os.environ.get("MEDGEMMA_USE_REAL_MODELS", "false").lower()
        self.use_real_models: bool = env_val in ("true", "1", "yes")

        if self.use_real_models:
            logger.info("ModelFactory: REAL model mode enabled")
        else:
            logger.info("ModelFactory: MOCK model mode (default)")

    def create_text_model(
        self,
        model_id: str = DEFAULT_TEXT_MODEL_ID,
    ) -> Any:
        """
        Create a text model (MedGemma 27B).

        In mock mode: returns a callable MagicMock.
        In real mode: loads with 4-bit NF4 quantization via transformers.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            A model instance (real or mock).
        """
        if not self.use_real_models:
            logger.info(f"Creating mock text model for '{model_id}'")
            return MagicMock()

        logger.info(f"Loading real text model '{model_id}' with quantization")
        return self._load_real_text_model(model_id)

    def create_vision_model(
        self,
        model_id: str = DEFAULT_VISION_MODEL_ID,
    ) -> Any:
        """
        Create a vision model (MedGemma 4B multimodal).

        In mock mode: returns a callable MagicMock.
        In real mode: loads with bfloat16 on single GPU.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            A model/pipeline instance (real or mock).
        """
        if not self.use_real_models:
            logger.info(f"Creating mock vision model for '{model_id}'")
            return MagicMock()

        logger.info(f"Loading real vision model '{model_id}' with bfloat16")
        return self._load_real_vision_model(model_id)

    def _load_real_text_model(self, model_id: str) -> Any:
        """
        Internal: Load the real 27B text model with quantization.
        Isolated for mocking in tests.

        Uses transformers AutoModelForCausalLM with BitsAndBytesConfig
        for 4-bit NF4 quantization and device_map="auto" for tensor
        parallelism across 2xT4 GPUs.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()
        model_kwargs = get_model_kwargs(qconfig, model_type="text")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        logger.info(f"Loaded text model '{model_id}' successfully")
        return model

    def _load_real_vision_model(self, model_id: str) -> Any:
        """
        Internal: Load the real 4B vision model with bfloat16.
        Isolated for mocking in tests.

        Uses transformers pipeline("image-text-to-text") with
        torch_dtype=bfloat16 on a single T4 GPU.
        """
        import torch
        from transformers import pipeline  # type: ignore

        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info(f"Loaded vision model '{model_id}' successfully")
        return pipe
