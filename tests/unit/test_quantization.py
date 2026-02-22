"""
Tests for quantization configuration utilities.

TDD: Written BEFORE src/utils/quantization.py.
Per plan: BitsAndBytesConfig with NF4 4-bit quantization for MedGemma 27B
on Kaggle 2xT4 GPUs (16GB VRAM each).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestQuantizationConfig:
    """Tests for the QuantizationConfig dataclass."""

    def test_default_config_is_4bit(self):
        """Default QuantizationConfig should use 4-bit quantization."""
        from utils.quantization import QuantizationConfig

        config = QuantizationConfig()
        assert config.load_in_4bit is True
        assert config.load_in_8bit is False

    def test_default_quant_type_is_nf4(self):
        """Default quant type should be NF4 (normalized float 4)."""
        from utils.quantization import QuantizationConfig

        config = QuantizationConfig()
        assert config.bnb_4bit_quant_type == "nf4"

    def test_default_compute_dtype_is_bfloat16(self):
        """Default compute dtype should be bfloat16."""
        from utils.quantization import QuantizationConfig

        config = QuantizationConfig()
        assert config.bnb_4bit_compute_dtype == "bfloat16"

    def test_config_allows_8bit_override(self):
        """Should support switching to 8-bit quantization."""
        from utils.quantization import QuantizationConfig

        config = QuantizationConfig(load_in_4bit=False, load_in_8bit=True)
        assert config.load_in_4bit is False
        assert config.load_in_8bit is True


class TestDetectGpuConfig:
    """Tests for GPU detection utility."""

    def test_detect_gpu_config_returns_dict(self):
        """detect_gpu_config() should return a dict with gpu info."""
        from utils.quantization import detect_gpu_config

        with patch("utils.quantization._get_gpu_count", return_value=2):
            with patch("utils.quantization._get_gpu_memory_gb", return_value=16.0):
                config = detect_gpu_config()

        assert isinstance(config, dict)
        assert "gpu_count" in config
        assert "gpu_memory_gb" in config
        assert config["gpu_count"] == 2
        assert config["gpu_memory_gb"] == 16.0

    def test_get_gpu_memory_gb_with_mocked_torch(self):
        """_get_gpu_memory_gb must use props.total_memory (not total_mem)."""
        from utils.quantization import _get_gpu_memory_gb

        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024 ** 3)  # 16 GB in bytes

        with patch("utils.quantization.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            result = _get_gpu_memory_gb()

        assert result == pytest.approx(16.0)
        mock_torch.cuda.get_device_properties.assert_called_once_with(0)

    def test_detect_gpu_config_no_gpu(self):
        """Should handle no GPU gracefully."""
        from utils.quantization import detect_gpu_config

        with patch("utils.quantization._get_gpu_count", return_value=0):
            with patch("utils.quantization._get_gpu_memory_gb", return_value=0.0):
                config = detect_gpu_config()

        assert config["gpu_count"] == 0


class TestCheckBitsandbytes:
    """Tests for the _check_bitsandbytes() early import verification."""

    def test_check_bitsandbytes_succeeds_when_importable(self):
        """_check_bitsandbytes() should be a no-op when bitsandbytes imports fine."""
        import sys
        from utils.quantization import _check_bitsandbytes

        mock_bnb = MagicMock()
        with patch.dict(sys.modules, {"bitsandbytes": mock_bnb}):
            # Should not raise
            _check_bitsandbytes()

    def test_check_bitsandbytes_raises_runtime_error_on_import_failure(self):
        """_check_bitsandbytes() should raise RuntimeError with real cause on failure."""
        import sys
        import builtins
        from utils.quantization import _check_bitsandbytes

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "bitsandbytes":
                raise ImportError("libcuda.so not found")
            return original_import(name, *args, **kwargs)

        # Ensure bitsandbytes is NOT cached in sys.modules so import is attempted
        modules_without_bnb = {k: v for k, v in sys.modules.items() if k != "bitsandbytes"}
        with patch.dict(sys.modules, modules_without_bnb, clear=True):
            with patch.object(builtins, "__import__", side_effect=failing_import):
                with pytest.raises(RuntimeError, match="libcuda.so not found"):
                    _check_bitsandbytes()

    def test_check_bitsandbytes_error_message_includes_diagnostic_hint(self):
        """RuntimeError message should include diagnostic guidance."""
        import sys
        from utils.quantization import _check_bitsandbytes

        with patch.dict(sys.modules, {"bitsandbytes": None}):
            with pytest.raises(RuntimeError, match="CUDA"):
                _check_bitsandbytes()


class TestGetBnbConfig:
    """Tests for BitsAndBytesConfig generation."""

    def test_get_bnb_config_returns_config_object(self):
        """get_bnb_config() should return a BitsAndBytesConfig-compatible dict."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="bfloat16"):
                    bnb = get_bnb_config(qconfig)

        MockBnB.assert_called_once()
        assert bnb is not None

    def test_get_bnb_config_passes_4bit_params(self):
        """get_bnb_config() should pass correct 4-bit params to BitsAndBytesConfig."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="bfloat16"):
                    get_bnb_config(qconfig)

        call_kwargs = MockBnB.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"

    def test_get_bnb_config_calls_check_bitsandbytes_first(self):
        """get_bnb_config() should call _check_bitsandbytes() before creating config."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes") as mock_check:
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="bfloat16"):
                    get_bnb_config(qconfig)

        mock_check.assert_called_once()

    def test_get_bnb_config_overrides_compute_dtype_for_t4(self):
        """On T4 GPUs (CC 7.5), get_bnb_config() should override bnb_4bit_compute_dtype to float16.

        The QuantizationConfig dataclass default is bfloat16, but T4 GPUs lack native
        bfloat16 tensor cores. Using bfloat16 compute dtype for 4-bit dequantization
        produces inf/nan logits, triggering a CUDA device-side assert.
        """
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()  # default bnb_4bit_compute_dtype="bfloat16"

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="float16"):
                    with patch("utils.quantization.torch") as mock_torch:
                        mock_torch.bfloat16 = "torch.bfloat16"
                        mock_torch.float16 = "torch.float16"
                        mock_torch.float32 = "torch.float32"
                        get_bnb_config(qconfig)

        call_kwargs = MockBnB.call_args[1]
        # On T4, compute dtype MUST be float16, NOT bfloat16
        assert call_kwargs["bnb_4bit_compute_dtype"] == "torch.float16"

    def test_get_bnb_config_keeps_bfloat16_compute_dtype_on_ampere(self):
        """On Ampere+ GPUs (CC >= 8.0), get_bnb_config() should keep bfloat16 compute dtype."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()  # default bnb_4bit_compute_dtype="bfloat16"

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="bfloat16"):
                    with patch("utils.quantization.torch") as mock_torch:
                        mock_torch.bfloat16 = "torch.bfloat16"
                        mock_torch.float16 = "torch.float16"
                        mock_torch.float32 = "torch.float32"
                        get_bnb_config(qconfig)

        call_kwargs = MockBnB.call_args[1]
        assert call_kwargs["bnb_4bit_compute_dtype"] == "torch.bfloat16"

    def test_get_bnb_config_logs_dtype_override(self):
        """When overriding compute dtype, get_bnb_config() should log the change."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()  # default bfloat16

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization._get_optimal_torch_dtype", return_value="float16"):
                    with patch("utils.quantization.torch") as mock_torch:
                        mock_torch.bfloat16 = "torch.bfloat16"
                        mock_torch.float16 = "torch.float16"
                        mock_torch.float32 = "torch.float32"
                        with patch("utils.quantization.logger") as mock_logger:
                            get_bnb_config(qconfig)

        # Should log a warning about overriding
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "float16" in warning_msg.lower() or "override" in warning_msg.lower()


class TestGetOptimalTorchDtype:
    """Tests for GPU-aware torch dtype selection."""

    def test_returns_bfloat16_for_ampere_gpu(self):
        """On Ampere+ GPUs (CC >= 8.0), should return 'bfloat16'."""
        from utils.quantization import _get_optimal_torch_dtype

        with patch("utils.quantization._get_gpu_compute_capability", return_value=(8, 0)):
            assert _get_optimal_torch_dtype() == "bfloat16"

    def test_returns_float16_for_t4_gpu(self):
        """On T4 (Turing, CC 7.5), should return 'float16' (no native bf16)."""
        from utils.quantization import _get_optimal_torch_dtype

        with patch("utils.quantization._get_gpu_compute_capability", return_value=(7, 5)):
            assert _get_optimal_torch_dtype() == "float16"

    def test_returns_float16_when_no_gpu(self):
        """Without GPU, should return 'float16' as safe default."""
        from utils.quantization import _get_optimal_torch_dtype

        with patch("utils.quantization._get_gpu_compute_capability", return_value=(0, 0)):
            assert _get_optimal_torch_dtype() == "float16"

    def test_returns_bfloat16_for_hopper_gpu(self):
        """On Hopper GPUs (CC 9.0), should return 'bfloat16'."""
        from utils.quantization import _get_optimal_torch_dtype

        with patch("utils.quantization._get_gpu_compute_capability", return_value=(9, 0)):
            assert _get_optimal_torch_dtype() == "bfloat16"


class TestGetGpuComputeCapability:
    """Tests for GPU compute capability detection."""

    def test_returns_capability_tuple_with_mocked_torch(self):
        """Should return (major, minor) compute capability."""
        from utils.quantization import _get_gpu_compute_capability

        with patch("utils.quantization.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_capability.return_value = (7, 5)
            result = _get_gpu_compute_capability()

        assert result == (7, 5)

    def test_returns_zero_tuple_when_no_gpu(self):
        """Should return (0, 0) when no GPU is available."""
        from utils.quantization import _get_gpu_compute_capability

        with patch("utils.quantization.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            result = _get_gpu_compute_capability()

        assert result == (0, 0)

    def test_returns_zero_tuple_when_torch_unavailable(self):
        """Should return (0, 0) when torch is None (not installed)."""
        from utils.quantization import _get_gpu_compute_capability

        # Patch torch to be None to simulate import failure
        with patch("utils.quantization.torch", None):
            result = _get_gpu_compute_capability()

        assert result == (0, 0)


class TestGetDeviceMap:
    """Tests for device map generation."""

    def test_get_device_map_dual_gpu(self):
        """For 2 GPUs, should return 'auto' device map."""
        from utils.quantization import get_device_map

        device_map = get_device_map(gpu_count=2)
        assert device_map == "auto"

    def test_get_device_map_single_gpu(self):
        """For 1 GPU, should return 'auto' device map."""
        from utils.quantization import get_device_map

        device_map = get_device_map(gpu_count=1)
        assert device_map == "auto"

    def test_get_device_map_no_gpu(self):
        """For 0 GPUs, should return 'cpu' device map."""
        from utils.quantization import get_device_map

        device_map = get_device_map(gpu_count=0)
        assert device_map == "cpu"


class TestGetModelKwargs:
    """Tests for model loading kwargs generation."""

    def test_get_model_kwargs_includes_device_map(self):
        """get_model_kwargs() should include device_map."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                    kwargs = get_model_kwargs(qconfig)

        assert "device_map" in kwargs
        assert kwargs["device_map"] == "auto"

    def test_get_model_kwargs_includes_quantization_config(self):
        """get_model_kwargs() should include quantization_config."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                mock_bnb = MagicMock()
                MockBnB.return_value = mock_bnb
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                    kwargs = get_model_kwargs(qconfig)

        assert "quantization_config" in kwargs
        assert kwargs["quantization_config"] is mock_bnb

    def test_get_model_kwargs_includes_max_memory_for_dual_t4(self):
        """get_model_kwargs() should include max_memory for 2 GPUs with ~16GB each."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                    kwargs = get_model_kwargs(qconfig)

        assert "max_memory" in kwargs
        # Kaggle T4s have ~15-16GB usable; we budget 14GiB to leave headroom
        assert kwargs["max_memory"] == {0: "14GiB", 1: "14GiB"}

    def test_get_model_kwargs_includes_torch_dtype(self):
        """get_model_kwargs() should include dtype."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                    with patch("utils.quantization._get_optimal_torch_dtype", return_value="bfloat16"):
                        with patch("utils.quantization.torch") as mock_torch:
                            mock_torch.bfloat16 = "torch.bfloat16"
                            kwargs = get_model_kwargs(qconfig)

        assert "dtype" in kwargs
        assert kwargs["dtype"] == "torch.bfloat16"

    def test_get_model_kwargs_torch_dtype_matches_optimal(self):
        """dtype should match _get_optimal_torch_dtype result."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                    # T4 GPU should get float16
                    with patch("utils.quantization._get_optimal_torch_dtype", return_value="float16"):
                        with patch("utils.quantization.torch") as mock_torch:
                            mock_torch.float16 = "torch.float16"
                            kwargs = get_model_kwargs(qconfig)

        assert kwargs["dtype"] == "torch.float16"

    def test_get_model_kwargs_includes_torch_dtype_for_transformers_compat(self):
        """get_model_kwargs should include torch_dtype for broader transformers compatibility.

        Some Kaggle images pin transformers builds that still expect torch_dtype.
        Supplying both avoids silent dtype fallback that can trigger NaN logits.
        """
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                    with patch("utils.quantization._get_optimal_torch_dtype", return_value="float16"):
                        with patch("utils.quantization.torch") as mock_torch:
                            mock_torch.float16 = "torch.float16"
                            kwargs = get_model_kwargs(qconfig)

        assert "torch_dtype" in kwargs
        assert kwargs["torch_dtype"] == "torch.float16"

    def test_get_model_kwargs_includes_attention_impl_for_text_models(self):
        """Text model kwargs should force eager attention for numerical stability."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                    kwargs = get_model_kwargs(qconfig, model_type="text")

        assert kwargs["attn_implementation"] == "eager"

    def test_get_model_kwargs_does_not_force_attention_impl_for_vision_models(self):
        """Vision kwargs should not set text attention implementation overrides."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                    kwargs = get_model_kwargs(qconfig, model_type="vision")

        assert "attn_implementation" not in kwargs

    def test_get_model_kwargs_keeps_dtype_key(self):
        """get_model_kwargs should continue returning dtype for newer transformers."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization._check_bitsandbytes"):
            with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
                MockBnB.return_value = MagicMock()
                with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                    with patch("utils.quantization._get_optimal_torch_dtype", return_value="float16"):
                        with patch("utils.quantization.torch") as mock_torch:
                            mock_torch.float16 = "torch.float16"
                            kwargs = get_model_kwargs(qconfig)

        assert "dtype" in kwargs
