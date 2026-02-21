"""
ModelFactory: Centralized model creation with feature flag.

Reads MEDGEMMA_USE_REAL_MODELS env var to decide whether to return
MockModelWrapper instances or real wrapped models for inference.

Usage:
    factory = ModelFactory()
    text_model = factory.create_text_model()  # Returns MockModelWrapper by default
    vision_model = factory.create_vision_model()

    # To use real models (requires GPU + model weights):
    # export MEDGEMMA_USE_REAL_MODELS=true

All returned models conform to the agent callable interface:
    Text:   result = model(prompt, max_tokens=N) -> {"choices": [{"text": "..."}]}
    Vision: result = model(images=..., prompt=...) -> [{"generated_text": "..."}]
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default model IDs
DEFAULT_TEXT_MODEL_ID = "google/medgemma-27b-text-it"
DEFAULT_VISION_MODEL_ID = "google/medgemma-4b-it"


def _verify_quantization(model: Any, model_id: str) -> None:
    """
    Verify that a model was actually quantized after loading.

    Logs a warning if quantization appears to have silently failed,
    which would cause CUDA OOM on limited-VRAM GPUs like T4.

    Args:
        model: The loaded model object.
        model_id: Model identifier for log messages.
    """
    is_quantized = getattr(model, "is_quantized", None)

    if is_quantized is None:
        logger.warning(
            f"Cannot verify quantization for '{model_id}': model has no "
            f"'is_quantized' attribute. If you see OOM errors, quantization "
            f"may not be applied. Check bitsandbytes CUDA compatibility."
        )
        return

    if is_quantized:
        logger.info(f"Model '{model_id}' loaded and verified as quantized")
    else:
        logger.warning(
            f"Model '{model_id}' is NOT quantized despite quantization_config "
            f"being provided. This will cause OOM on T4 GPUs. Check that "
            f"bitsandbytes CUDA kernels are functional: "
            f"python -c 'import bitsandbytes; print(bitsandbytes.__version__)'"
        )


class ModelFactory:
    """
    Factory for creating text and vision models.

    Controlled by MEDGEMMA_USE_REAL_MODELS env var:
    - False (default): Returns MockModelWrapper instances for testing.
    - True: Loads real quantized models wrapped in TextModelWrapper/VisionModelWrapper.
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

        In mock mode: returns a MockModelWrapper(mode="text").
        In real mode: loads with 4-bit NF4 quantization, wraps in TextModelWrapper.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            A callable model wrapper conforming to the agent interface.
        """
        if not self.use_real_models:
            from utils.model_wrappers import MockModelWrapper

            logger.info(f"Creating mock text model for '{model_id}'")
            return MockModelWrapper(mode="text")

        logger.info(f"Loading real text model '{model_id}' with quantization")
        return self._load_real_text_model(model_id)

    def create_vision_model(
        self,
        model_id: str = DEFAULT_VISION_MODEL_ID,
    ) -> Any:
        """
        Create a vision model (MedGemma 4B multimodal).

        In mock mode: returns a MockModelWrapper(mode="vision").
        In real mode: loads with bfloat16, wraps in VisionModelWrapper.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            A callable model wrapper conforming to the agent interface.
        """
        if not self.use_real_models:
            from utils.model_wrappers import MockModelWrapper

            logger.info(f"Creating mock vision model for '{model_id}'")
            return MockModelWrapper(mode="vision")

        logger.info(f"Loading real vision model '{model_id}' with bfloat16")
        return self._load_real_vision_model(model_id)

    def _load_real_text_model(self, model_id: str) -> Any:
        """
        Internal: Load the real 27B text model with quantization and wrap it.
        Isolated for mocking in tests.

        Uses transformers AutoModelForCausalLM with BitsAndBytesConfig
        for 4-bit NF4 quantization and device_map="auto" for tensor
        parallelism across 2xT4 GPUs.

        Returns a TextModelWrapper for agent-compatible calling convention.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from utils.model_wrappers import TextModelWrapper
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()
        model_kwargs = get_model_kwargs(qconfig, model_type="text")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        # Verify quantization was actually applied (catches silent bnb failures)
        _verify_quantization(model, model_id)

        logger.info(f"Loaded text model '{model_id}' successfully, wrapping")
        return TextModelWrapper(model=model, tokenizer=tokenizer)

    def _load_real_vision_model(self, model_id: str) -> Any:
        """
        Internal: Load the real 4B vision model with bfloat16 and wrap it.
        Isolated for mocking in tests.

        Uses transformers pipeline("image-text-to-text") with
        torch_dtype=bfloat16 on a single T4 GPU.

        Returns a VisionModelWrapper for agent-compatible calling convention.
        """
        import torch
        from transformers import pipeline  # type: ignore

        from utils.model_wrappers import VisionModelWrapper

        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info(f"Loaded vision model '{model_id}' successfully, wrapping")
        return VisionModelWrapper(pipeline=pipe)
