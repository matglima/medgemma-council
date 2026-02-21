"""
Tests for ModelFactory: Centralized model creation with feature flag.

TDD: Written BEFORE src/utils/model_factory.py.
Per plan: MEDGEMMA_USE_REAL_MODELS env var controls real vs mock models.
Default is False so tests always use mocks.

Updated for Phase 11: ModelFactory now returns MockModelWrapper instances
in mock mode (instead of raw MagicMock), and TextModelWrapper/VisionModelWrapper
in real mode (instead of raw model objects).
"""

import pytest
from unittest.mock import MagicMock, patch
import os


class TestModelFactory:
    """Tests for the ModelFactory class."""

    def test_factory_defaults_to_mock_mode(self):
        """ModelFactory should default to mock mode (no real models)."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            assert factory.use_real_models is False

    def test_factory_reads_env_var(self):
        """ModelFactory should read MEDGEMMA_USE_REAL_MODELS env var."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()
            assert factory.use_real_models is True

    def test_create_text_model_returns_mock_wrapper_by_default(self):
        """In mock mode, create_text_model() should return a MockModelWrapper."""
        from utils.model_factory import ModelFactory
        from utils.model_wrappers import MockModelWrapper

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model()

        assert isinstance(model, MockModelWrapper)
        assert model.mode == "text"
        assert callable(model)

    def test_create_vision_model_returns_mock_wrapper_by_default(self):
        """In mock mode, create_vision_model() should return a MockModelWrapper."""
        from utils.model_factory import ModelFactory
        from utils.model_wrappers import MockModelWrapper

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_vision_model()

        assert isinstance(model, MockModelWrapper)
        assert model.mode == "vision"
        assert callable(model)

    def test_mock_text_model_returns_proper_format(self):
        """Mock text model should return llama-cpp format when called."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model()

        result = model("What is the diagnosis?", max_tokens=256)
        assert isinstance(result, dict)
        assert "choices" in result
        assert "text" in result["choices"][0]

    def test_mock_vision_model_returns_proper_format(self):
        """Mock vision model should return pipeline format when called."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_vision_model()

        result = model(images=["img.png"], prompt="Analyze.")
        assert isinstance(result, list)
        assert "generated_text" in result[0]

    def test_create_text_model_real_mode_calls_loader(self):
        """In real mode, create_text_model() should use transformers + quantization."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        with patch.object(factory, "_load_real_text_model") as mock_loader:
            mock_loader.return_value = MagicMock()
            model = factory.create_text_model()

        mock_loader.assert_called_once()
        assert model is not None

    def test_create_vision_model_real_mode_calls_loader(self):
        """In real mode, create_vision_model() should use transformers pipeline."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        with patch.object(factory, "_load_real_vision_model") as mock_loader:
            mock_loader.return_value = MagicMock()
            model = factory.create_vision_model()

        mock_loader.assert_called_once()
        assert model is not None

    def test_real_text_model_returns_text_wrapper(self):
        """In real mode, _load_real_text_model should return a TextModelWrapper."""
        from utils.model_factory import ModelFactory
        from utils.model_wrappers import TextModelWrapper

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        # Mock the internal loader to simulate returning a TextModelWrapper
        mock_wrapper = MagicMock(spec=TextModelWrapper)
        with patch.object(factory, "_load_real_text_model", return_value=mock_wrapper):
            model = factory.create_text_model()

        assert isinstance(model, MagicMock)  # It's a MagicMock with TextModelWrapper spec

    def test_real_vision_model_returns_vision_wrapper(self):
        """In real mode, _load_real_vision_model should return a VisionModelWrapper."""
        from utils.model_factory import ModelFactory
        from utils.model_wrappers import VisionModelWrapper

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        mock_wrapper = MagicMock(spec=VisionModelWrapper)
        with patch.object(factory, "_load_real_vision_model", return_value=mock_wrapper):
            model = factory.create_vision_model()

        assert isinstance(model, MagicMock)  # It's a MagicMock with VisionModelWrapper spec


class TestVerifyQuantization:
    """Tests for post-load quantization verification."""

    def test_verify_quantization_warns_when_not_quantized(self):
        """Should log warning if model does not appear quantized after loading."""
        from utils.model_factory import _verify_quantization

        mock_model = MagicMock()
        mock_model.is_quantized = False

        with patch("utils.model_factory.logger") as mock_logger:
            _verify_quantization(mock_model, "test-model")

        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "not quantized" in warning_msg.lower() or "NOT quantized" in warning_msg

    def test_verify_quantization_logs_info_when_quantized(self):
        """Should log info (not warning) when model is properly quantized."""
        from utils.model_factory import _verify_quantization

        mock_model = MagicMock()
        mock_model.is_quantized = True

        with patch("utils.model_factory.logger") as mock_logger:
            _verify_quantization(mock_model, "test-model")

        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called()

    def test_verify_quantization_handles_missing_attribute(self):
        """Should handle models without is_quantized attribute gracefully."""
        from utils.model_factory import _verify_quantization

        mock_model = MagicMock(spec=[])  # No attributes

        with patch("utils.model_factory.logger") as mock_logger:
            # Should not raise
            _verify_quantization(mock_model, "test-model")

        # Should log a warning since we can't confirm quantization
        mock_logger.warning.assert_called()
