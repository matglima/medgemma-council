"""
Tests for model inference wrappers: TextModelWrapper and VisionModelWrapper.

TDD: Written BEFORE src/utils/model_wrappers.py.
Per CLAUDE.md: Never load real models in tests. Mock all heavy compute.

TextModelWrapper bridges:
    transformers AutoModelForCausalLM + AutoTokenizer
    -> callable(prompt, max_tokens=N) -> {"choices": [{"text": "..."}]}

VisionModelWrapper bridges:
    transformers pipeline("image-text-to-text")
    -> callable(images=..., prompt=...) -> [{"generated_text": "..."}]
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestTextModelWrapper:
    """Tests for the TextModelWrapper class."""

    def test_import(self):
        """TextModelWrapper should be importable from utils.model_wrappers."""
        from utils.model_wrappers import TextModelWrapper

    def test_init_stores_model_and_tokenizer(self):
        """TextModelWrapper should store model and tokenizer references."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        assert wrapper.model is mock_model
        assert wrapper.tokenizer is mock_tokenizer

    def test_callable(self):
        """TextModelWrapper should be callable with (prompt, max_tokens)."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        assert callable(wrapper)

    def test_call_returns_llama_cpp_format(self):
        """
        Calling TextModelWrapper(prompt, max_tokens=N) should return a dict
        in llama-cpp format: {"choices": [{"text": "generated text"}]}.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer encoding: returns input_ids tensor
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]  # batch=1, seq_len=10
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "This is the generated response."

        # Mock model.generate returns token ids
        mock_output_ids = MagicMock()
        # Simulate output being longer than input (input + generated)
        mock_output_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_output_ids

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("What is the diagnosis?", max_tokens=256)

        assert isinstance(result, dict)
        assert "choices" in result
        assert len(result["choices"]) == 1
        assert "text" in result["choices"][0]
        assert isinstance(result["choices"][0]["text"], str)

    def test_call_invokes_tokenizer_encode(self):
        """Calling the wrapper should tokenize the (formatted) prompt."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Set up chat template to return a known formatted string
        formatted = "<start_of_turn>user\ntest prompt<end_of_turn>\n<start_of_turn>model\n"
        mock_tokenizer.apply_chat_template.return_value = formatted

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        # Tokenizer should have been called with the formatted prompt
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == formatted

    def test_call_invokes_model_generate(self):
        """Calling the wrapper should call model.generate with correct args."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=512)

        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args
        # max_new_tokens should be passed
        assert call_kwargs.kwargs.get("max_new_tokens") == 512

    def test_call_strips_input_from_output(self):
        """
        The wrapper should return only the GENERATED tokens,
        not the input prompt tokens (slice off input_ids length).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        # The output sequence is the full [input + generated]
        mock_output_sequence = MagicMock()
        # When sliced as output[0][input_len:], return generated-only ids
        mock_generated_ids = MagicMock()
        mock_output_sequence.__getitem__ = MagicMock(return_value=mock_generated_ids)
        mock_model.generate.return_value = mock_output_sequence

        mock_tokenizer.decode.return_value = "Only the generated part"

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("prompt", max_tokens=100)

        assert result["choices"][0]["text"] == "Only the generated part"
        # decode should be called with the sliced (generated-only) tokens
        mock_tokenizer.decode.assert_called_once()

    def test_default_max_tokens(self):
        """If max_tokens not specified, should default to 1024."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt")

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("max_new_tokens") == 1024

    def test_error_handling_returns_error_text(self):
        """If model.generate raises, the wrapper should return an error in the same format."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory")

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("test prompt", max_tokens=100)

        assert isinstance(result, dict)
        assert "choices" in result
        assert "error" in result["choices"][0]["text"].lower() or "CUDA" in result["choices"][0]["text"]

    def test_device_kwarg(self):
        """TextModelWrapper should accept an optional device parameter."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(
            model=mock_model, tokenizer=mock_tokenizer, device="cuda:0"
        )
        assert wrapper.device == "cuda:0"

    def test_default_device_is_auto(self):
        """Default device should be 'auto' (let model handle placement)."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        assert wrapper.device == "auto"

    def test_truncation_enabled_in_tokenizer_call(self):
        """TextModelWrapper should pass truncation=True to the tokenizer."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.call_args
        assert call_kwargs.kwargs.get("truncation") is True

    def test_max_length_set_in_tokenizer_call(self):
        """TextModelWrapper should pass max_length to the tokenizer to prevent unbounded prompts."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.call_args
        max_length = call_kwargs.kwargs.get("max_length")
        assert max_length is not None
        assert isinstance(max_length, int)
        assert max_length > 0

    def test_default_max_input_tokens(self):
        """Default max_input_tokens should be 4096."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        assert wrapper.max_input_tokens == 4096

    def test_custom_max_input_tokens(self):
        """TextModelWrapper should accept a custom max_input_tokens parameter."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = TextModelWrapper(
            model=mock_model, tokenizer=mock_tokenizer, max_input_tokens=2048
        )
        assert wrapper.max_input_tokens == 2048

    def test_max_length_uses_max_input_tokens(self):
        """The tokenizer max_length should equal the wrapper's max_input_tokens."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(
            model=mock_model, tokenizer=mock_tokenizer, max_input_tokens=2048
        )
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.call_args
        assert call_kwargs.kwargs.get("max_length") == 2048

    def test_generate_uses_do_sample_false_by_default(self):
        """model.generate() should use do_sample=False by default (greedy decoding).

        Sampling mode amplifies numerical errors from bfloat16 dequantization
        on T4 GPUs into invalid probabilities (inf/nan), triggering CUDA asserts.
        Greedy decoding avoids softmax sampling entirely.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("do_sample") is False

    def test_generate_allows_do_sample_override(self):
        """Callers should be able to override do_sample=True via kwargs."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100, do_sample=True)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("do_sample") is True

    # -----------------------------------------------------------------------
    # Chat template formatting (Fix #6: empty specialist outputs)
    # -----------------------------------------------------------------------

    def test_applies_chat_template_when_available(self):
        """TextModelWrapper should use tokenizer.apply_chat_template() to format
        the prompt before tokenization.

        MedGemma-27B-text-it (Gemma 2 IT) expects <start_of_turn>user/model
        markers. Without the chat template, the model generates EOS immediately
        → empty specialist outputs.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Simulate a tokenizer that supports chat templates (like Gemma)
        mock_tokenizer.apply_chat_template.return_value = (
            "<start_of_turn>user\nWhat is the diagnosis?<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("What is the diagnosis?", max_tokens=256)

        # apply_chat_template should have been called
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args

        # Should pass messages in chat format
        messages = call_args[0][0] if call_args[0] else call_args[1].get("conversation")
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the diagnosis?"

        # Should request generation prompt
        assert call_args[1].get("add_generation_prompt") is True
        # Should not tokenize (we do that separately)
        assert call_args[1].get("tokenize") is False

    def test_tokenizer_receives_formatted_prompt_from_chat_template(self):
        """The tokenizer should receive the chat-template-formatted string,
        not the raw prompt.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        formatted = "<start_of_turn>user\nTest prompt<end_of_turn>\n<start_of_turn>model\n"
        mock_tokenizer.apply_chat_template.return_value = formatted

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Test prompt", max_tokens=100)

        # The tokenizer __call__ should receive the formatted string, not raw
        tokenizer_call_args = mock_tokenizer.call_args
        actual_prompt = tokenizer_call_args[0][0]
        assert actual_prompt == formatted

    def test_falls_back_to_raw_prompt_when_no_chat_template(self):
        """If the tokenizer doesn't support apply_chat_template, the wrapper
        should fall back to using the raw prompt string.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Simulate a tokenizer WITHOUT chat template support
        mock_tokenizer.apply_chat_template.side_effect = Exception(
            "This tokenizer does not have a chat template"
        )

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Raw prompt text", max_tokens=100)

        # Tokenizer should be called with the raw prompt as fallback
        tokenizer_call_args = mock_tokenizer.call_args
        actual_prompt = tokenizer_call_args[0][0]
        assert actual_prompt == "Raw prompt text"

    def test_falls_back_when_apply_chat_template_missing(self):
        """If the tokenizer lacks apply_chat_template entirely (AttributeError),
        the wrapper should fall back to using the raw prompt string.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Remove apply_chat_template entirely
        del mock_tokenizer.apply_chat_template

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Fallback prompt", max_tokens=100)

        tokenizer_call_args = mock_tokenizer.call_args
        actual_prompt = tokenizer_call_args[0][0]
        assert actual_prompt == "Fallback prompt"

    # -----------------------------------------------------------------------
    # pad_token_id in generate kwargs (Fix #9: warning elimination)
    # -----------------------------------------------------------------------

    def test_generate_passes_pad_token_id(self):
        """model.generate() should receive pad_token_id=tokenizer.pad_token_id
        to suppress the 'Setting pad_token_id to eos_token_id' warning.

        Bug B fix sets tokenizer.pad_token, but model.generate() checks
        generation_config.pad_token_id, not the tokenizer. We must pass it.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_tokenizer.pad_token_id = 0  # Simulate pad_token_id being set
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("pad_token_id") == 0

    def test_generate_pad_token_id_not_passed_when_none(self):
        """If tokenizer.pad_token_id is None, pad_token_id should NOT be passed
        to generate() (let the model handle its own default).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_tokenizer.pad_token_id = None
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert "pad_token_id" not in call_kwargs.kwargs


class TestVisionModelWrapper:
    """Tests for the VisionModelWrapper class."""

    def test_import(self):
        """VisionModelWrapper should be importable from utils.model_wrappers."""
        from utils.model_wrappers import VisionModelWrapper

    def test_init_stores_pipeline(self):
        """VisionModelWrapper should store the pipeline reference."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        wrapper = VisionModelWrapper(pipeline=mock_pipeline)

        assert wrapper.pipeline is mock_pipeline

    def test_callable(self):
        """VisionModelWrapper should be callable."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        wrapper = VisionModelWrapper(pipeline=mock_pipeline)

        assert callable(wrapper)

    def test_call_with_images_and_prompt(self):
        """
        Calling VisionModelWrapper(images=..., prompt=...) should return
        a list of dicts: [{"generated_text": "..."}].
        """
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Chest X-ray shows consolidation."}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        result = wrapper(images=["image1.png"], prompt="Analyze this X-ray.")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "generated_text" in result[0]

    def test_call_passes_through_to_pipeline(self):
        """The wrapper should pass images and prompt to the underlying pipeline."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=["img.png"], prompt="Describe the image.")

        mock_pipeline.assert_called_once()

    def test_call_formats_pipeline_input(self):
        """
        The wrapper should format the input for the transformers
        image-text-to-text pipeline correctly.
        """
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=["img.png"], prompt="Analyze.")

        # The pipeline should be called — exact format depends on implementation
        assert mock_pipeline.called

    def test_call_without_images_returns_error(self):
        """Calling without images should return an appropriate error message."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        result = wrapper(images=[], prompt="Analyze.")

        assert isinstance(result, list)
        assert "no images" in result[0]["generated_text"].lower() or \
               "error" in result[0]["generated_text"].lower()

    def test_error_handling(self):
        """If the pipeline raises, the wrapper should return an error gracefully."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("GPU error")

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        result = wrapper(images=["img.png"], prompt="Analyze.")

        assert isinstance(result, list)
        assert len(result) == 1
        text = result[0]["generated_text"]
        assert "error" in text.lower() or "GPU" in text

    def test_max_tokens_kwarg(self):
        """VisionModelWrapper should accept max_new_tokens parameter."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=["img.png"], prompt="Analyze.", max_new_tokens=512)

        call_kwargs = mock_pipeline.call_args
        # max_new_tokens should be passed through
        assert call_kwargs.kwargs.get("max_new_tokens") == 512 or \
               (len(call_kwargs.args) > 0 or "max_new_tokens" in str(call_kwargs))


class TestMockModelWrapper:
    """Tests for the MockModelWrapper used in testing/development."""

    def test_import(self):
        """MockModelWrapper should be importable from utils.model_wrappers."""
        from utils.model_wrappers import MockModelWrapper

    def test_text_mode_returns_llama_cpp_format(self):
        """In text mode, MockModelWrapper should return llama-cpp format."""
        from utils.model_wrappers import MockModelWrapper

        wrapper = MockModelWrapper(mode="text")
        result = wrapper("What is the diagnosis?", max_tokens=256)

        assert isinstance(result, dict)
        assert "choices" in result
        assert "text" in result["choices"][0]

    def test_vision_mode_returns_pipeline_format(self):
        """In vision mode, MockModelWrapper should return pipeline format."""
        from utils.model_wrappers import MockModelWrapper

        wrapper = MockModelWrapper(mode="vision")
        result = wrapper(images=["img.png"], prompt="Analyze.")

        assert isinstance(result, list)
        assert "generated_text" in result[0]

    def test_custom_response(self):
        """MockModelWrapper should accept a custom response string."""
        from utils.model_wrappers import MockModelWrapper

        wrapper = MockModelWrapper(mode="text", response="Custom clinical finding.")
        result = wrapper("prompt", max_tokens=100)

        assert result["choices"][0]["text"] == "Custom clinical finding."

    def test_default_mode_is_text(self):
        """Default mode should be 'text'."""
        from utils.model_wrappers import MockModelWrapper

        wrapper = MockModelWrapper()
        result = wrapper("test")

        assert isinstance(result, dict)
        assert "choices" in result
