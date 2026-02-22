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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default model IDs
# Default to MedGemma 1.5 4B for faster/stabler Kaggle inference.
# Advanced users can override to 27B via MEDGEMMA_TEXT_MODEL_ID.
DEFAULT_TEXT_MODEL_ID = "google/medgemma-1.5-4b-it"
DEFAULT_VISION_MODEL_ID = "google/medgemma-1.5-4b-it"

# Backward-compatible aliases
_MODEL_ID_ALIASES = {
    "google/medgemma-4b-it": "google/medgemma-1.5-4b-it",
}


def _normalize_model_id(model_id: str) -> str:
    """Normalize known legacy model IDs to canonical IDs."""
    normalized = _MODEL_ID_ALIASES.get(model_id, model_id)
    if normalized != model_id:
        logger.warning(
            f"ModelFactory: model_id '{model_id}' is deprecated; "
            f"using '{normalized}'"
        )
    return normalized


def _is_default_4b_model(model_id: str) -> bool:
    """Return True for canonical MedGemma 1.5 4B IDs."""
    return "medgemma-1.5-4b" in model_id.lower()


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

    Model Caching:
        Uses a class-level cache so that models are loaded only once per
        (type, model_id) pair. This prevents CUDA OOM when multiple graph
        nodes each create a ModelFactory and call create_text_model() —
        instead of loading the text model N times, they all share one instance.

        Cache keys are prefixed with "text:" or "vision:" to keep namespaces
        separate even if the same model_id string is used for both.

        Call ModelFactory.clear_cache() to reset (used in tests for isolation).
    """

    # Class-level model cache: {"text:<model_id>": wrapper, "vision:<model_id>": wrapper}
    _model_cache: Dict[str, Any] = {}

    def __init__(self) -> None:
        env_val = os.environ.get("MEDGEMMA_USE_REAL_MODELS", "false").lower()
        self.use_real_models: bool = env_val in ("true", "1", "yes")

        if self.use_real_models:
            logger.info("ModelFactory: REAL model mode enabled")
        else:
            logger.info("ModelFactory: MOCK model mode (default)")

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the class-level model cache.

        Intended for test isolation — ensures each test starts with a
        fresh cache. In production, models stay cached for the lifetime
        of the process.
        """
        cls._model_cache.clear()
        logger.debug("ModelFactory: model cache cleared")

    def create_text_model(
        self,
        model_id: Optional[str] = None,
    ) -> Any:
        """
        Create or retrieve a cached text model.

        In mock mode: returns a cached MockModelWrapper(mode="text").
        In real mode:
        - default 4B model loads in non-quantized fp16 mode for Kaggle stability
        - optional larger models (e.g., 27B) load with 4-bit quantization
          via QuantizationConfig/get_model_kwargs.

        Models are cached by (type, model_id) so repeated calls with the same
        model_id return the same object — critical for avoiding CUDA OOM when
        multiple graph nodes each instantiate a ModelFactory.

        Model ID resolution order:
            1. Explicit ``model_id`` argument (if provided)
            2. ``MEDGEMMA_TEXT_MODEL_ID`` environment variable
            3. ``DEFAULT_TEXT_MODEL_ID`` constant

        Args:
            model_id: HuggingFace model ID. When *None*, falls back to the
                ``MEDGEMMA_TEXT_MODEL_ID`` env var, then the built-in default.

        Returns:
            A callable model wrapper conforming to the agent interface.
        """
        if model_id is None:
            model_id = os.environ.get(
                "MEDGEMMA_TEXT_MODEL_ID", DEFAULT_TEXT_MODEL_ID
            )
            logger.debug(
                f"Resolved text model_id to '{model_id}' "
                f"(env var {'set' if 'MEDGEMMA_TEXT_MODEL_ID' in os.environ else 'not set'})"
            )

        model_id = _normalize_model_id(model_id)

        cache_key = f"text:{model_id}"

        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Cache hit for text model '{model_id}'")
            return ModelFactory._model_cache[cache_key]

        # Reuse a cached 4B vision pipeline when available to avoid loading
        # a second copy of the same model on GPU.
        if self.use_real_models and _is_default_4b_model(model_id):
            vision_cache_key = f"vision:{model_id}"
            cached_vision = ModelFactory._model_cache.get(vision_cache_key)
            shared_pipeline = getattr(cached_vision, "pipeline", None)
            if shared_pipeline is not None:
                from utils.model_wrappers import PipelineTextModelWrapper

                logger.info(
                    f"Reusing cached 4B vision pipeline for text model '{model_id}'"
                )
                wrapper = PipelineTextModelWrapper(pipeline=shared_pipeline)
                ModelFactory._model_cache[cache_key] = wrapper
                return wrapper

        if not self.use_real_models:
            from utils.model_wrappers import MockModelWrapper

            logger.info(f"Creating mock text model for '{model_id}'")
            wrapper = MockModelWrapper(mode="text")
        else:
            logger.info(f"Loading real text model '{model_id}' for inference")
            wrapper = self._load_real_text_model(model_id)

        ModelFactory._model_cache[cache_key] = wrapper
        return wrapper

    def create_vision_model(
        self,
        model_id: str = DEFAULT_VISION_MODEL_ID,
    ) -> Any:
        """
        Create or retrieve a cached vision model (MedGemma 4B multimodal).

        In mock mode: returns a cached MockModelWrapper(mode="vision").
        In real mode: loads with bfloat16, wraps in VisionModelWrapper.

        Args:
            model_id: HuggingFace model ID.

        Returns:
            A callable model wrapper conforming to the agent interface.
        """
        model_id = _normalize_model_id(model_id)
        cache_key = f"vision:{model_id}"

        if cache_key in ModelFactory._model_cache:
            logger.debug(f"Cache hit for vision model '{model_id}'")
            return ModelFactory._model_cache[cache_key]

        # Reuse a cached 4B text pipeline when available to avoid loading
        # a second copy of the same model on GPU.
        if self.use_real_models and _is_default_4b_model(model_id):
            text_cache_key = f"text:{model_id}"
            cached_text = ModelFactory._model_cache.get(text_cache_key)
            shared_pipeline = getattr(cached_text, "pipeline", None)
            if shared_pipeline is not None:
                from utils.model_wrappers import VisionModelWrapper

                logger.info(
                    f"Reusing cached 4B text pipeline for vision model '{model_id}'"
                )
                wrapper = VisionModelWrapper(pipeline=shared_pipeline)
                ModelFactory._model_cache[cache_key] = wrapper
                return wrapper

        if not self.use_real_models:
            from utils.model_wrappers import MockModelWrapper

            logger.info(f"Creating mock vision model for '{model_id}'")
            wrapper = MockModelWrapper(mode="vision")
        else:
            logger.info(f"Loading real vision model '{model_id}' with bfloat16")
            wrapper = self._load_real_vision_model(model_id)

        ModelFactory._model_cache[cache_key] = wrapper
        return wrapper

    def _load_real_text_model(self, model_id: str) -> Any:
        """
        Internal: Load a real text model and wrap it.
        Isolated for mocking in tests.

        Loading strategy:
        - 4B default (`google/medgemma-1.5-4b-it`): official
          pipeline("image-text-to-text") path
        - larger text models (e.g., 27B): AutoModel + 4-bit NF4 quantization
          kwargs

        Returns a text-capable wrapper for the agent calling convention.
        """
        is_default_4b_text = _is_default_4b_model(model_id)

        if is_default_4b_text:
            from transformers import pipeline  # type: ignore

            from utils.model_wrappers import PipelineTextModelWrapper

            try:
                import torch  # type: ignore
            except ImportError:
                torch = None  # type: ignore

            if torch is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                use_bf16 = False
                if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
                    use_bf16 = bool(torch.cuda.is_bf16_supported())
                dtype: Any = torch.bfloat16 if use_bf16 else torch.float16
            else:
                device = "cpu"
                dtype = "float16"

            pipe_kwargs: Dict[str, Any] = {
                "model": model_id,
                "device": device,
                "dtype": dtype,
                "trust_remote_code": True,
            }

            # Keep compatibility across transformers versions.
            try:
                pipe = pipeline("image-text-to-text", **pipe_kwargs)
            except TypeError:
                pipe_kwargs.pop("dtype", None)
                pipe_kwargs["torch_dtype"] = dtype
                try:
                    pipe = pipeline("image-text-to-text", **pipe_kwargs)
                except TypeError:
                    pipe_kwargs.pop("trust_remote_code", None)
                    pipe = pipeline("image-text-to-text", **pipe_kwargs)

            logger.info(
                f"Loaded default 4B text pipeline '{model_id}' "
                f"with dtype={dtype}, device={device}"
            )
            return PipelineTextModelWrapper(pipeline=pipe)

        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from utils.model_wrappers import TextModelWrapper
        from utils.quantization import QuantizationConfig, get_model_kwargs

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Ensure critical token IDs are set for proper generation.
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "<pad>"
            logger.info(
                f"Set tokenizer.pad_token = '{tokenizer.pad_token}' "
                f"(pad_token_id={tokenizer.pad_token_id}) for model '{model_id}'"
            )

        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 1
            logger.warning(
                "tokenizer.eos_token_id was None, set to 1 (Gemma default)"
            )

        qconfig = QuantizationConfig()
        model_kwargs = get_model_kwargs(qconfig, model_type="text")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        if hasattr(model, "generation_config"):
            if model.generation_config.eos_token_id is None:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
                logger.info(
                    f"Set model.generation_config.eos_token_id = {tokenizer.eos_token_id}"
                )
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
                logger.info(
                    f"Set model.generation_config.pad_token_id = {tokenizer.pad_token_id}"
                )
            if model.generation_config.bos_token_id is None:
                model.generation_config.bos_token_id = tokenizer.bos_token_id
                logger.info(
                    f"Set model.generation_config.bos_token_id = {tokenizer.bos_token_id}"
                )

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
