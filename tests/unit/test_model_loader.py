"""
Tests for model loader / VRAM management utilities.

TDD: Written BEFORE src/utils/model_loader.py.
Per MASTER_PROMPT: Must handle strict VRAM management. Load/unload models
dynamically or offload layers to CPU RAM where possible.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestModelLoader:
    """Tests for the ModelLoader VRAM management class."""

    def test_loader_initializes_with_empty_registry(self):
        """ModelLoader starts with no models loaded."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        assert loader.loaded_models == {}

    def test_load_text_model_registers_it(self):
        """Loading a text model (llama-cpp) should register it by name."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        with patch("utils.model_loader.ModelLoader._load_llama_cpp") as mock_load:
            mock_load.return_value = MagicMock()
            model = loader.load_text_model("medgemma-27b", model_path="/fake/path.gguf")
            assert "medgemma-27b" in loader.loaded_models
            assert model is not None

    def test_load_vision_model_registers_it(self):
        """Loading a vision model (transformers pipeline) should register it."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        with patch("utils.model_loader.ModelLoader._load_transformers_pipeline") as mock_load:
            mock_load.return_value = MagicMock()
            model = loader.load_vision_model("medgemma-4b", model_id="google/medgemma-4b")
            assert "medgemma-4b" in loader.loaded_models

    def test_unload_model_frees_registry(self):
        """Unloading a model should remove it from the registry."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        loader.loaded_models["test-model"] = MagicMock()
        loader.unload_model("test-model")
        assert "test-model" not in loader.loaded_models

    def test_unload_nonexistent_model_is_safe(self):
        """Unloading a model that isn't loaded should not raise."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        loader.unload_model("nonexistent")  # Should not raise

    def test_get_model_returns_loaded_model(self):
        """get_model should return a previously loaded model by name."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        mock_model = MagicMock()
        loader.loaded_models["test-model"] = mock_model
        assert loader.get_model("test-model") is mock_model

    def test_get_model_returns_none_for_unloaded(self):
        """get_model should return None if the model is not loaded."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        assert loader.get_model("not-loaded") is None

    def test_swap_model_unloads_then_loads(self):
        """swap_model should unload old model before loading new one (VRAM constraint)."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()
        old_model = MagicMock()
        loader.loaded_models["old-model"] = old_model

        with patch("utils.model_loader.ModelLoader._load_llama_cpp") as mock_load:
            mock_load.return_value = MagicMock()
            loader.swap_model(
                unload_name="old-model",
                load_name="new-model",
                model_path="/fake/new.gguf",
                model_type="text",
            )
            assert "old-model" not in loader.loaded_models
            assert "new-model" in loader.loaded_models


class TestQuantizedModelLoading:
    """Tests for quantized model loading methods (Phase 9)."""

    def test_load_text_model_quantized_registers_model(self):
        """load_text_model_quantized() should load via transformers + bnb and register."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()

        with patch.object(loader, "_load_transformers_quantized") as mock_load:
            mock_load.return_value = MagicMock()
            model = loader.load_text_model_quantized(
                name="medgemma-27b-q4",
                model_id="google/medgemma-27b-text-it",
            )

        assert "medgemma-27b-q4" in loader.loaded_models
        assert model is not None
        mock_load.assert_called_once()

    def test_load_vision_model_quantized_registers_model(self):
        """load_vision_model_quantized() should load with bfloat16 and register."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()

        with patch.object(loader, "_load_transformers_vision_bf16") as mock_load:
            mock_load.return_value = MagicMock()
            model = loader.load_vision_model_quantized(
                name="medgemma-4b-vision",
                model_id="google/medgemma-1.5-4b-it",
            )

        assert "medgemma-4b-vision" in loader.loaded_models
        assert model is not None
        mock_load.assert_called_once()

    def test_get_memory_usage_returns_dict(self):
        """get_memory_usage() should return a dict with per-GPU memory info."""
        from utils.model_loader import ModelLoader

        loader = ModelLoader()

        with patch("utils.model_loader._get_torch_memory_stats") as mock_stats:
            mock_stats.return_value = {"gpu_0_used_gb": 5.2, "gpu_1_used_gb": 4.8, "total_used_gb": 10.0}
            usage = loader.get_memory_usage()

        assert isinstance(usage, dict)
        assert "total_used_gb" in usage
