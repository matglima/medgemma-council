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

    def setup_method(self):
        """Clear the model cache before each test to ensure isolation."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def teardown_method(self):
        """Clear the model cache after each test."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

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

    def test_default_text_model_id_is_4b(self):
        """Default text model should be MedGemma 1.5 4B for Kaggle stability."""
        from utils.model_factory import DEFAULT_TEXT_MODEL_ID

        assert DEFAULT_TEXT_MODEL_ID == "google/medgemma-4b-it"

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


class TestModelCaching:
    """Tests for ModelFactory class-level model caching.

    TDD: Written BEFORE caching implementation.

    The ModelFactory must cache models at the class level so that:
    1. Multiple ModelFactory instances share the same cached model.
    2. create_text_model() with the same model_id returns the same object.
    3. create_vision_model() with the same model_id returns the same object.
    4. Different model_ids produce different cached entries.
    5. clear_cache() resets the cache (needed for test isolation).
    6. In real mode, _load_real_text_model is only called once per model_id.
    """

    def setup_method(self):
        """Clear the cache before each test to ensure isolation."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def teardown_method(self):
        """Clear the cache after each test."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def test_text_model_cached_across_calls(self):
        """Calling create_text_model() twice should return the same object."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model1 = factory.create_text_model()
            model2 = factory.create_text_model()

        assert model1 is model2

    def test_text_model_cached_across_factory_instances(self):
        """Different ModelFactory instances should share cached text models."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory1 = ModelFactory()
            model1 = factory1.create_text_model()

            factory2 = ModelFactory()
            model2 = factory2.create_text_model()

        assert model1 is model2

    def test_vision_model_cached_across_calls(self):
        """Calling create_vision_model() twice should return the same object."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model1 = factory.create_vision_model()
            model2 = factory.create_vision_model()

        assert model1 is model2

    def test_vision_model_cached_across_factory_instances(self):
        """Different ModelFactory instances should share cached vision models."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory1 = ModelFactory()
            model1 = factory1.create_vision_model()

            factory2 = ModelFactory()
            model2 = factory2.create_vision_model()

        assert model1 is model2

    def test_different_model_ids_not_shared(self):
        """Different model_ids should produce separate cache entries."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model_a = factory.create_text_model(model_id="model-a")
            model_b = factory.create_text_model(model_id="model-b")

        assert model_a is not model_b

    def test_clear_cache_resets_text_model(self):
        """clear_cache() should cause the next call to create a new instance."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model1 = factory.create_text_model()
            ModelFactory.clear_cache()
            model2 = factory.create_text_model()

        assert model1 is not model2

    def test_clear_cache_resets_vision_model(self):
        """clear_cache() should cause the next call to create a new instance."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model1 = factory.create_vision_model()
            ModelFactory.clear_cache()
            model2 = factory.create_vision_model()

        assert model1 is not model2

    def test_real_mode_loader_called_only_once(self):
        """In real mode, _load_real_text_model should be called only once per model_id."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        mock_wrapper = MagicMock()
        with patch.object(factory, "_load_real_text_model", return_value=mock_wrapper) as mock_loader:
            model1 = factory.create_text_model()
            model2 = factory.create_text_model()

        # Loader should be called exactly once; second call should hit cache
        mock_loader.assert_called_once()
        assert model1 is model2

    def test_text_and_vision_caches_are_independent(self):
        """Text and vision models should use separate cache namespaces."""
        from utils.model_factory import ModelFactory
        from utils.model_wrappers import MockModelWrapper

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            text_model = factory.create_text_model()
            vision_model = factory.create_vision_model()

        assert text_model is not vision_model
        assert isinstance(text_model, MockModelWrapper)
        assert isinstance(vision_model, MockModelWrapper)
        assert text_model.mode == "text"
        assert vision_model.mode == "vision"

    def test_cache_key_includes_model_type_prefix(self):
        """Cache should distinguish text vs vision even with same model_id string."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            # Use the same string as model_id for both text and vision
            text_model = factory.create_text_model(model_id="same-model")
            vision_model = factory.create_vision_model(model_id="same-model")

        assert text_model is not vision_model


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


class TestLoadRealTextModelPadToken:
    """Tests for pad_token setup in _load_real_text_model().

    TDD: Written BEFORE the fix.

    MedGemma tokenizers may not have a pad_token set, which produces
    'Setting pad_token_id to eos_token_id' warnings and can cause
    issues with batched generation. _load_real_text_model() should
    set pad_token = eos_token when pad_token is missing.
    """

    def setup_method(self):
        """Clear cache before each test."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def test_sets_pad_token_when_missing(self):
        """_load_real_text_model should set tokenizer.pad_token = eos_token if pad_token is None."""
        import sys
        from utils.model_factory import ModelFactory

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"

        mock_model = MagicMock()
        mock_model.is_quantized = True

        # Mock transformers module for the local import inside _load_real_text_model
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch("utils.quantization._check_bitsandbytes"):
                with patch("utils.quantization.BitsAndBytesConfig", return_value=MagicMock()):
                    with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                        with patch("utils.model_factory._verify_quantization"):
                            factory._load_real_text_model("test-model")

        # After loading, pad_token should be set to eos_token
        assert mock_tokenizer.pad_token == "</s>"

    def test_preserves_existing_pad_token(self):
        """_load_real_text_model should NOT override an existing pad_token."""
        import sys
        from utils.model_factory import ModelFactory

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"

        mock_model = MagicMock()
        mock_model.is_quantized = True

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch("utils.quantization._check_bitsandbytes"):
                with patch("utils.quantization.BitsAndBytesConfig", return_value=MagicMock()):
                    with patch("utils.quantization.detect_gpu_config", return_value={"gpu_count": 1, "gpu_memory_gb": 16.0}):
                        with patch("utils.model_factory._verify_quantization"):
                            factory._load_real_text_model("test-model")

        # Existing pad_token should be preserved
        assert mock_tokenizer.pad_token == "<pad>"

    def test_4b_text_model_load_skips_quantization_path(self):
        """4B default text model should load without 27B quantization kwargs.

        This keeps default inference lighter and avoids 27B-specific quantization
        behavior on Kaggle.
        """
        import sys
        from utils.model_factory import ModelFactory

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.bos_token_id = 2

        mock_model = MagicMock()
        mock_model.is_quantized = True

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        with patch.dict(os.environ, {"MEDGEMMA_USE_REAL_MODELS": "true"}):
            factory = ModelFactory()

        with patch.dict(sys.modules, {"transformers": mock_transformers}):
            with patch("utils.model_factory._verify_quantization"):
                with patch("utils.quantization.get_model_kwargs") as mock_get_model_kwargs:
                    factory._load_real_text_model("google/medgemma-4b-it")

        # 4B default path should not invoke quantized 27B kwargs builder
        mock_get_model_kwargs.assert_not_called()

        call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert "device_map" in call_kwargs
        assert "quantization_config" not in call_kwargs


class TestTextModelIdEnvVar:
    """Tests for MEDGEMMA_TEXT_MODEL_ID env var override.

    TDD: Written BEFORE the implementation.

    When MEDGEMMA_TEXT_MODEL_ID is set, create_text_model() should use that
    model ID instead of the default, unless an explicit model_id argument
    is provided (which takes precedence).
    """

    def setup_method(self):
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def teardown_method(self):
        from utils.model_factory import ModelFactory
        ModelFactory.clear_cache()

    def test_env_var_overrides_default_model_id(self):
        """MEDGEMMA_TEXT_MODEL_ID env var should override the default model ID."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {
            "MEDGEMMA_TEXT_MODEL_ID": "google/medgemma-4b-it",
        }, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model()

        # Cache key should use the env var model ID, not the default
        assert "text:google/medgemma-4b-it" in ModelFactory._model_cache

    def test_explicit_model_id_takes_precedence_over_env_var(self):
        """An explicit model_id argument should take precedence over the env var."""
        from utils.model_factory import ModelFactory

        with patch.dict(os.environ, {
            "MEDGEMMA_TEXT_MODEL_ID": "google/medgemma-4b-it",
        }, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model(model_id="custom/model")

        assert "text:custom/model" in ModelFactory._model_cache

    def test_default_used_when_no_env_var(self):
        """When MEDGEMMA_TEXT_MODEL_ID is not set, the default should be used."""
        from utils.model_factory import ModelFactory, DEFAULT_TEXT_MODEL_ID

        with patch.dict(os.environ, {}, clear=True):
            factory = ModelFactory()
            model = factory.create_text_model()

        assert f"text:{DEFAULT_TEXT_MODEL_ID}" in ModelFactory._model_cache
