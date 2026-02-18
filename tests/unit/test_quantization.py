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

    def test_detect_gpu_config_no_gpu(self):
        """Should handle no GPU gracefully."""
        from utils.quantization import detect_gpu_config

        with patch("utils.quantization._get_gpu_count", return_value=0):
            with patch("utils.quantization._get_gpu_memory_gb", return_value=0.0):
                config = detect_gpu_config()

        assert config["gpu_count"] == 0


class TestGetBnbConfig:
    """Tests for BitsAndBytesConfig generation."""

    def test_get_bnb_config_returns_config_object(self):
        """get_bnb_config() should return a BitsAndBytesConfig-compatible dict."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()

        with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
            MockBnB.return_value = MagicMock()
            bnb = get_bnb_config(qconfig)

        MockBnB.assert_called_once()
        assert bnb is not None

    def test_get_bnb_config_passes_4bit_params(self):
        """get_bnb_config() should pass correct 4-bit params to BitsAndBytesConfig."""
        from utils.quantization import QuantizationConfig, get_bnb_config

        qconfig = QuantizationConfig()

        with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
            MockBnB.return_value = MagicMock()
            get_bnb_config(qconfig)

        call_kwargs = MockBnB.call_args[1]
        assert call_kwargs["load_in_4bit"] is True
        assert call_kwargs["bnb_4bit_quant_type"] == "nf4"


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

        with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
            mock_bnb = MagicMock()
            MockBnB.return_value = mock_bnb
            with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                kwargs = get_model_kwargs(qconfig)

        assert "quantization_config" in kwargs
        assert kwargs["quantization_config"] is mock_bnb

    def test_get_model_kwargs_includes_max_memory_for_dual_t4(self):
        """For dual T4s, should set max_memory to 14GiB each for headroom."""
        from utils.quantization import QuantizationConfig, get_model_kwargs

        qconfig = QuantizationConfig()

        with patch("utils.quantization.BitsAndBytesConfig") as MockBnB:
            MockBnB.return_value = MagicMock()
            with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 2, "gpu_memory_gb": 16.0}):
                kwargs = get_model_kwargs(qconfig)

        assert "max_memory" in kwargs
        assert kwargs["max_memory"] == {0: "14GiB", 1: "14GiB"}
