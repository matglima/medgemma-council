"""
Tests for ModelFactory: Centralized model creation with feature flag.

TDD: Written BEFORE src/utils/model_factory.py.
Per plan: MEDGEMMA_USE_REAL_MODELS env var controls real vs mock models.
Default is False so tests always use mocks.
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

    def test_create_text_model_returns_mock_by_default(self):
        """In mock mode, create_text_model() should return a MagicMock."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model()

        assert model is not None
        # Should be callable (like a real model)
        assert callable(model)

    def test_create_vision_model_returns_mock_by_default(self):
        """In mock mode, create_vision_model() should return a MagicMock."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_vision_model()

        assert model is not None
        assert callable(model)

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
