"""
Quantization configuration for MedGemma models on Kaggle 2xT4 GPUs.

Provides:
- QuantizationConfig: Dataclass for 4-bit/8-bit quantization settings.
- detect_gpu_config(): Detect available GPU hardware.
- get_bnb_config(): Generate a BitsAndBytesConfig for transformers.
- get_device_map(): Determine device_map strategy based on GPU count.
- get_model_kwargs(): Build complete kwargs dict for model loading.

Hardware targets:
- MedGemma 27B text: 4-bit NF4 quantization across 2xT4 (~13.5GB weights)
- MedGemma 4B vision: bfloat16 on single T4 (~8GB)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Lazy import for environments without bitsandbytes
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:
    BitsAndBytesConfig = None  # type: ignore


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Defaults to 4-bit NF4 quantization with bfloat16 compute dtype,
    optimized for Kaggle 2xT4 GPUs (16GB VRAM each).
    """

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


def _get_gpu_count() -> int:
    """Get the number of available GPUs. Isolated for mocking."""
    try:
        import torch
        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 0


def _get_gpu_memory_gb() -> float:
    """Get memory of first GPU in GB. Isolated for mocking."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        return 0.0
    except (ImportError, RuntimeError):
        return 0.0


def _get_gpu_compute_capability() -> tuple:
    """
    Get compute capability of first GPU as (major, minor) tuple.

    Returns (0, 0) if no GPU available or torch not installed.
    Used to select optimal dtype — Ampere+ (8.0+) supports native bfloat16,
    older GPUs like T4 (7.5) should use float16.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
        return (0, 0)
    except (ImportError, RuntimeError):
        return (0, 0)


def _get_optimal_torch_dtype() -> str:
    """
    Select optimal torch dtype based on GPU compute capability.

    - Ampere+ (CC >= 8.0): bfloat16 (native hardware support)
    - Turing/T4 (CC < 8.0) or no GPU: float16 (safe default)

    This affects both the BitsAndBytesConfig compute dtype and the
    torch_dtype passed to from_pretrained() for non-quantized parameters.
    """
    major, _ = _get_gpu_compute_capability()
    if major >= 8:
        return "bfloat16"
    return "float16"


def detect_gpu_config() -> Dict[str, Any]:
    """
    Detect available GPU hardware configuration.

    Returns:
        Dict with 'gpu_count' and 'gpu_memory_gb' keys.
    """
    gpu_count = _get_gpu_count()
    gpu_memory_gb = _get_gpu_memory_gb()

    config = {
        "gpu_count": gpu_count,
        "gpu_memory_gb": round(gpu_memory_gb, 1),
    }

    logger.info(f"Detected GPU config: {config}")
    return config


def _check_bitsandbytes() -> None:
    """
    Verify that bitsandbytes can actually be imported with its CUDA backend.

    transformers will silently catch import failures and raise a misleading
    "pip install bitsandbytes" error. This function surfaces the real cause
    (e.g., missing CUDA libraries, version mismatch) early.

    Raises:
        RuntimeError: If bitsandbytes cannot be imported, with the real error.
    """
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"bitsandbytes is installed but failed to import: {e}\n"
            "This usually means CUDA libraries are missing or incompatible.\n"
            "Check that your CUDA toolkit version matches your bitsandbytes build.\n"
            "Try: python -c 'import bitsandbytes' to see the full traceback."
        ) from e


def get_bnb_config(qconfig: QuantizationConfig) -> Any:
    """
    Generate a BitsAndBytesConfig from our QuantizationConfig.

    Args:
        qconfig: Our quantization configuration dataclass.

    Returns:
        A transformers BitsAndBytesConfig instance.

    Raises:
        RuntimeError: If bitsandbytes cannot be imported (CUDA issues).
    """
    # Verify bitsandbytes is actually functional before transformers tries
    # to use it (transformers gives a misleading "pip install" error otherwise)
    _check_bitsandbytes()

    # Resolve compute dtype — defer torch import for test-friendliness
    compute_dtype = qconfig.bnb_4bit_compute_dtype
    try:
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(qconfig.bnb_4bit_compute_dtype, torch.bfloat16)
    except ImportError:
        # In test environments without torch, pass string (BitsAndBytesConfig is mocked)
        pass

    return BitsAndBytesConfig(
        load_in_4bit=qconfig.load_in_4bit,
        load_in_8bit=qconfig.load_in_8bit,
        bnb_4bit_quant_type=qconfig.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qconfig.bnb_4bit_use_double_quant,
    )


def get_device_map(gpu_count: int) -> Union[str, Dict[str, Any]]:
    """
    Determine the device_map strategy based on available GPUs.

    Args:
        gpu_count: Number of available GPUs.

    Returns:
        'auto' for GPU setups (lets accelerate handle distribution),
        'cpu' for CPU-only.
    """
    if gpu_count == 0:
        return "cpu"
    return "auto"


def get_model_kwargs(
    qconfig: QuantizationConfig,
    model_type: str = "text",
) -> Dict[str, Any]:
    """
    Build complete kwargs dict for model loading with quantization.

    Args:
        qconfig: Quantization configuration.
        model_type: 'text' for 27B text model, 'vision' for 4B multimodal.

    Returns:
        Dict of kwargs suitable for AutoModelForCausalLM.from_pretrained().
    """
    gpu_config = detect_gpu_config()
    gpu_count = gpu_config["gpu_count"]

    device_map = get_device_map(gpu_count)
    bnb_config = get_bnb_config(qconfig)

    # Resolve dtype for non-quantized parameters (embeddings, layernorm).
    # Without this, from_pretrained() may load params in float32, wasting memory.
    # T4 (CC 7.5) uses float16; Ampere+ (CC 8.0+) uses bfloat16.
    # Note: We use 'dtype' (not deprecated 'torch_dtype') for newer transformers.
    optimal_dtype_str = _get_optimal_torch_dtype()
    torch_dtype = optimal_dtype_str  # fallback: pass string if torch unavailable
    try:
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(optimal_dtype_str, torch.float16)
    except ImportError:
        pass

    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "quantization_config": bnb_config,
        "dtype": torch_dtype,
    }

    # Set max_memory for multi-GPU setups (14GiB per T4 for headroom)
    if gpu_count >= 2:
        kwargs["max_memory"] = {
            i: "14GiB" for i in range(gpu_count)
        }

    logger.info(
        f"Model kwargs for {model_type}: device_map={device_map}, "
        f"gpu_count={gpu_count}, quantization={qconfig.bnb_4bit_quant_type}, "
        f"torch_dtype={optimal_dtype_str}"
    )

    return kwargs
