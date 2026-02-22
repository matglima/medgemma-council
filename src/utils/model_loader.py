"""
ModelLoader: VRAM management for MedGemma-Council.

Handles dynamic loading/unloading of models to stay within Kaggle
dual-T4 GPU constraints (16GB VRAM each). Supports:
- llama-cpp-python models (legacy text inference)
- transformers pipelines (MedGemma 1.5 4B multimodal)
- transformers + bitsandbytes quantized models (optional MedGemma 27B text)
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def _get_torch_memory_stats() -> Dict[str, float]:
    """
    Get per-GPU memory usage stats from PyTorch.
    Isolated for mocking in tests.

    Returns:
        Dict with gpu_N_used_gb keys and total_used_gb.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"total_used_gb": 0.0}

        stats: Dict[str, float] = {}
        total = 0.0
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / (1024 ** 3)
            stats[f"gpu_{i}_used_gb"] = round(used, 2)
            total += used
        stats["total_used_gb"] = round(total, 2)
        return stats
    except (ImportError, RuntimeError):
        return {"total_used_gb": 0.0}


class ModelLoader:
    """
    Manages GPU memory by tracking loaded models and supporting
    dynamic load/unload/swap operations.
    """

    def __init__(self) -> None:
        self.loaded_models: Dict[str, Any] = {}

    def load_text_model(self, name: str, model_path: str, **kwargs: Any) -> Any:
        """
        Load a text model via llama-cpp-python.

        Args:
            name: Registry key for the model.
            model_path: Path to the GGUF model file.
            **kwargs: Additional args passed to Llama constructor.

        Returns:
            The loaded model instance.
        """
        logger.info(f"Loading text model '{name}' from {model_path}")
        model = self._load_llama_cpp(model_path, **kwargs)
        self.loaded_models[name] = model
        return model

    def load_vision_model(self, name: str, model_id: str, **kwargs: Any) -> Any:
        """
        Load a vision model via transformers pipeline.

        Args:
            name: Registry key for the model.
            model_id: HuggingFace model ID or local path.
            **kwargs: Additional args passed to pipeline constructor.

        Returns:
            The loaded pipeline instance.
        """
        logger.info(f"Loading vision model '{name}' from {model_id}")
        model = self._load_transformers_pipeline(model_id, **kwargs)
        self.loaded_models[name] = model
        return model

    def unload_model(self, name: str) -> None:
        """
        Unload a model from the registry and free resources.

        Args:
            name: Registry key of the model to unload.
        """
        if name in self.loaded_models:
            model = self.loaded_models.pop(name)
            # Attempt to free GPU memory
            if hasattr(model, "close"):
                model.close()
            del model
            logger.info(f"Unloaded model '{name}'")
        else:
            logger.warning(f"Model '{name}' not found in registry, nothing to unload")

    def get_model(self, name: str) -> Optional[Any]:
        """
        Retrieve a loaded model by name.

        Args:
            name: Registry key.

        Returns:
            The model instance, or None if not loaded.
        """
        return self.loaded_models.get(name, None)

    def swap_model(
        self,
        unload_name: str,
        load_name: str,
        model_path: str,
        model_type: str = "text",
        **kwargs: Any,
    ) -> Any:
        """
        Atomically swap one model for another to manage VRAM.

        Unloads the old model first, then loads the new one.

        Args:
            unload_name: Registry key of the model to unload.
            load_name: Registry key for the new model.
            model_path: Path/ID of the new model.
            model_type: 'text' for llama-cpp, 'vision' for transformers.
            **kwargs: Additional args for model loading.

        Returns:
            The newly loaded model instance.
        """
        self.unload_model(unload_name)

        if model_type == "text":
            return self.load_text_model(load_name, model_path, **kwargs)
        elif model_type == "vision":
            return self.load_vision_model(load_name, model_path, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'text' or 'vision'.")

    def _load_llama_cpp(self, model_path: str, **kwargs: Any) -> Any:
        """
        Internal: Load a model via llama-cpp-python.
        Isolated for easy mocking in tests.
        """
        from llama_cpp import Llama  # type: ignore

        return Llama(model_path=model_path, **kwargs)

    def _load_transformers_pipeline(self, model_id: str, **kwargs: Any) -> Any:
        """
        Internal: Load a vision model via transformers pipeline.
        Isolated for easy mocking in tests.
        """
        from transformers import pipeline  # type: ignore

        return pipeline("image-text-to-text", model=model_id, **kwargs)

    def load_text_model_quantized(
        self,
        name: str,
        model_id: str,
        **kwargs: Any,
    ) -> Any:
        """
        Load a text model with 4-bit quantization via transformers + bitsandbytes.

        Uses NF4 quantization with bfloat16 compute dtype and device_map="auto"
        for automatic tensor parallelism across available GPUs.

        Args:
            name: Registry key for the model.
            model_id: HuggingFace model ID (e.g., "google/medgemma-27b-text-it"
                for optional larger-model runs).
            **kwargs: Additional args passed to from_pretrained().

        Returns:
            The loaded quantized model instance.
        """
        logger.info(f"Loading quantized text model '{name}' from {model_id}")
        model = self._load_transformers_quantized(model_id, **kwargs)
        self.loaded_models[name] = model
        return model

    def load_vision_model_quantized(
        self,
        name: str,
        model_id: str,
        **kwargs: Any,
    ) -> Any:
        """
        Load a vision model with bfloat16 precision.

        For MedGemma 4B multimodal: ~8GB at bfloat16, fits on single T4.

        Args:
            name: Registry key for the model.
            model_id: HuggingFace model ID (e.g., "google/medgemma-4b-it").
            **kwargs: Additional args passed to pipeline().

        Returns:
            The loaded vision pipeline instance.
        """
        logger.info(f"Loading vision model '{name}' from {model_id} (bfloat16)")
        model = self._load_transformers_vision_bf16(model_id, **kwargs)
        self.loaded_models[name] = model
        return model

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage across all devices.

        Returns:
            Dict with per-GPU usage (gpu_N_used_gb) and total_used_gb.
        """
        return _get_torch_memory_stats()

    def _load_transformers_quantized(self, model_id: str, **kwargs: Any) -> Any:
        """
        Internal: Load a model with 4-bit NF4 quantization.
        Isolated for easy mocking in tests.
        """
        from transformers import AutoModelForCausalLM  # type: ignore

        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()
        model_kwargs = get_model_kwargs(qconfig, model_type="text")
        model_kwargs.update(kwargs)

        return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def _load_transformers_vision_bf16(self, model_id: str, **kwargs: Any) -> Any:
        """
        Internal: Load a vision model with bfloat16 precision.
        Isolated for easy mocking in tests.
        """
        import torch
        from transformers import pipeline  # type: ignore

        return pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs,
        )
