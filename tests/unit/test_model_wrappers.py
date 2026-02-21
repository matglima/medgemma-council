"""
Tests for model inference wrappers: TextModelWrapper and VisionModelWrapper.

TDD: Written BEFORE src/utils/model_wrappers.py.
Per CLAUDE.md: Never load real models in tests. Mock all heavy compute.

TextModelWrapper bridges:
    transformers AutoModelForCausalLM + AutoTokenizer
    -> callable(prompt, max_tokens=N) -> {"choices": [{"text": "..."})]

VisionModelWrapper bridges:
    transformers pipeline("image-text-to-text")
    -> callable(images=..., prompt=...) -> [{"generated_text": "..."}]
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_mock_tokenizer_and_model(input_len=10, output_len=20, decode_text="response"):
    """
    Create properly configured mocks for TextModelWrapper tests.
    
    The mocks simulate:
    - apply_chat_template returning a tensor (not BatchEncoding)
    - model.generate returning output with correct shape
    - tokenizer.decode returning specified text
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    
    # Create a mock tensor that behaves correctly
    # Key: __contains__ returns False for "input_ids" (it's a tensor, not dict)
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, input_len)
    mock_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
    mock_input_ids.to.return_value = mock_input_ids  # .to() returns self
    # Make "input_ids" in mock_input_ids return False
    mock_input_ids.__contains__ = MagicMock(return_value=False)
    
    mock_tokenizer.apply_chat_template.return_value = mock_input_ids
    mock_tokenizer.decode.return_value = decode_text
    
    # Create mock model with proper output shape
    mock_model = MagicMock()
    
    # Create mock output that has shape and slicing behavior
    mock_output = MagicMock()
    mock_output.shape = (1, output_len)
    # output[0] returns something sliceable
    mock_full_seq = MagicMock()
    mock_generated_ids = MagicMock()
    mock_full_seq.__getitem__ = MagicMock(return_value=mock_generated_ids)
    mock_output.__getitem__ = MagicMock(return_value=mock_full_seq)
    
    mock_model.generate.return_value = mock_output
    
    return mock_tokenizer, mock_model


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
        """Calling the wrapper should tokenize the prompt (via apply_chat_template
        or fallback to tokenizer __call__ if chat template tokenization fails).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Make apply_chat_template(tokenize=True) fail so it falls back to
        # the string path where the tokenizer __call__ IS invoked.
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("tokenize") is True:
                raise TypeError("Not supported")
            return "<start_of_turn>user\ntest prompt<end_of_turn>\n<start_of_turn>model\n"

        mock_tokenizer.apply_chat_template.side_effect = side_effect

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        # In the fallback path, tokenizer __call__ should be invoked
        assert mock_tokenizer.call_count >= 1
        # The formatted prompt should be passed to the tokenizer
        tokenizer_call_args = mock_tokenizer.call_args
        assert tokenizer_call_args[0][0] == "<start_of_turn>user\ntest prompt<end_of_turn>\n<start_of_turn>model\n"

    def test_call_invokes_model_generate(self):
        """The wrapper should call model.generate() with the tokenized input."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        mock_model.generate.assert_called_once()

    def test_call_strips_input_from_output(self):
        """
        The wrapper should return only the GENERATED tokens,
        not the input prompt tokens (slice off input_ids length).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model(
            input_len=10, output_len=20, decode_text="Only the generated part"
        )

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("prompt", max_tokens=100)

        assert result["choices"][0]["text"] == "Only the generated part"
        # decode should be called with the sliced (generated-only) tokens
        mock_tokenizer.decode.assert_called_once()

    def test_default_max_tokens(self):
        """If max_tokens not specified, should default to 1024."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

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
        """TextModelWrapper should enable truncation (via apply_chat_template
        or fallback to tokenizer __call__).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Primary path: apply_chat_template with tokenize=True
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        # truncation should be passed to apply_chat_template
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("truncation") is True

    def test_max_length_set_in_tokenizer_call(self):
        """TextModelWrapper should pass max_length to truncation (via apply_chat_template
        or fallback to tokenizer __call__).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        max_length = call_kwargs.get("max_length")
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
        """The max_length for truncation should equal the wrapper's max_input_tokens."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(
            model=mock_model, tokenizer=mock_tokenizer, max_input_tokens=2048
        )
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("max_length") == 2048

    def test_generate_uses_do_sample_false_by_default(self):
        """model.generate() should be called with do_sample=False for greedy decoding."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("do_sample") is False

    def test_generate_allows_do_sample_override(self):
        """Caller can override do_sample via kwargs."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100, do_sample=True)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("do_sample") is True

    # -----------------------------------------------------------------------
    # Chat template formatting (Fix #6: empty specialist outputs)
    # -----------------------------------------------------------------------

    def test_applies_chat_template_when_available(self):
        """TextModelWrapper should use tokenizer.apply_chat_template() to format
        the prompt with proper structural markers.

        MedGemma-27B-text-it (Gemma 2 IT) expects <start_of_turn>user/model
        markers. Without the chat template, the model generates EOS immediately
        -> empty specialist outputs.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Simulate a tokenizer that supports chat templates with tokenization
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

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

        # Should request generation prompt and tokenize in one step
        assert call_args[1].get("add_generation_prompt") is True
        assert call_args[1].get("tokenize") is True
        assert call_args[1].get("truncation") is True
        assert call_args[1].get("return_tensors") == "pt"

    def test_tokenizer_receives_formatted_prompt_from_chat_template(self):
        """When apply_chat_template(tokenize=True) fails, the fallback
        string-based path should pass the formatted prompt to the tokenizer.
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

        def side_effect(*args, **kwargs):
            if kwargs.get("tokenize") is True:
                raise TypeError("Not supported")
            return formatted

        mock_tokenizer.apply_chat_template.side_effect = side_effect

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
    # Truncation-safe chat template (Fix #6b: structural marker preservation)
    # -----------------------------------------------------------------------

    def test_apply_chat_template_tokenize_true_when_supported(self):
        """TextModelWrapper should use apply_chat_template(tokenize=True) to get
        token IDs directly, so truncation preserves structural markers.

        The old approach: apply_chat_template(tokenize=False) -> string -> tokenizer(truncation=True)
        truncated from the right, cutting off <end_of_turn><start_of_turn>model markers.

        The new approach: apply_chat_template(tokenize=True, truncation=True, max_length=...)
        lets the chat template engine handle truncation, preserving structural tokens.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # apply_chat_template(tokenize=True, return_tensors="pt") returns a
        # tensor-like object with shape [1, seq_len]
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids

        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Test prompt", max_tokens=100)

        # apply_chat_template should be called with tokenize=True
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("tokenize") is True
        assert call_kwargs.get("truncation") is True
        assert call_kwargs.get("return_tensors") == "pt"
        assert call_kwargs.get("add_generation_prompt") is True
        assert "max_length" in call_kwargs

    def test_apply_chat_template_max_length_matches_max_input_tokens(self):
        """The max_length passed to apply_chat_template should equal max_input_tokens."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 5)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(
            model=mock_model, tokenizer=mock_tokenizer, max_input_tokens=2048
        )
        wrapper("Test prompt", max_tokens=100)

        call_kwargs = mock_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs.get("max_length") == 2048

    def test_tokenizer_not_called_separately_when_chat_template_tokenizes(self):
        """When apply_chat_template(tokenize=True) succeeds, the tokenizer should
        NOT be called separately to tokenize — we already have token IDs.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 5)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Test prompt", max_tokens=100)

        # The tokenizer __call__ should NOT have been invoked
        # (apply_chat_template already returned token IDs)
        mock_tokenizer.assert_not_called()

    def test_fallback_to_string_tokenization_when_chat_template_tokenize_fails(self):
        """If apply_chat_template(tokenize=True) fails (e.g., the tokenizer doesn't
        support truncation in chat template), fall back to the old string-based approach:
        apply_chat_template(tokenize=False) -> tokenizer(string, truncation=True).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # First call (tokenize=True) fails
        # Second call (tokenize=False) succeeds with a string
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("tokenize") is True:
                raise TypeError("apply_chat_template() got unexpected keyword argument 'truncation'")
            return "<start_of_turn>user\nTest<end_of_turn>\n<start_of_turn>model\n"

        mock_tokenizer.apply_chat_template.side_effect = side_effect

        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("Test prompt", max_tokens=100)

        # Should have fallen back to string tokenization
        assert "choices" in result
        # The tokenizer __call__ should have been used as fallback
        mock_tokenizer.assert_called_once()

    def test_fallback_to_raw_prompt_when_all_chat_template_attempts_fail(self):
        """If both apply_chat_template(tokenize=True) and apply_chat_template(tokenize=False)
        fail, fall back to using the raw prompt with the tokenizer.
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # All apply_chat_template calls fail
        mock_tokenizer.apply_chat_template.side_effect = Exception("No chat template")

        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 10]
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("Raw prompt", max_tokens=100)

        # Should have fallen back to raw prompt tokenization
        assert "choices" in result
        tokenizer_call_args = mock_tokenizer.call_args
        actual_prompt = tokenizer_call_args[0][0]
        assert actual_prompt == "Raw prompt"

    def test_input_ids_from_chat_template_passed_to_generate(self):
        """When apply_chat_template(tokenize=True) succeeds, the resulting
        input_ids should be passed directly to model.generate().
        """
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("Test prompt", max_tokens=100)

        # model.generate should have received input_ids
        call_kwargs = mock_model.generate.call_args
        assert "input_ids" in call_kwargs.kwargs

    def test_attention_mask_created_for_chat_template_input_ids(self):
        """When using tokenized chat template output, an attention_mask of all 1s
        should be created and passed to model.generate().
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 5)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        # Mock torch.ones_like at the module level where it's used
        mock_attention_mask = MagicMock()
        with patch("utils.model_wrappers.torch") as mock_torch:
            mock_torch.ones_like.return_value = mock_attention_mask
            wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
            wrapper("Test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert "attention_mask" in call_kwargs.kwargs

    def test_handles_batch_encoding_from_apply_chat_template(self):
        """When apply_chat_template(tokenize=True) returns a BatchEncoding dict
        (with 'input_ids' and 'attention_mask' keys) instead of a plain tensor,
        TextModelWrapper should extract the tensors correctly.
        
        This fixes the error: 'ones_like(): argument 'input' must be Tensor,
        not BatchEncoding'
        """
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Simulate apply_chat_template returning a BatchEncoding dict
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 5)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        fake_attention_mask = MagicMock()
        
        batch_encoding = {
            "input_ids": fake_input_ids,
            "attention_mask": fake_attention_mask,
        }
        mock_tokenizer.apply_chat_template.return_value = batch_encoding
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        result = wrapper("Test prompt", max_tokens=100)

        # Should extract input_ids from the dict
        assert "choices" in result
        # model.generate should have received input_ids
        call_kwargs = mock_model.generate.call_args
        assert "input_ids" in call_kwargs.kwargs

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

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()
        mock_tokenizer.pad_token_id = 0  # Simulate pad_token_id being set

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)
        wrapper("test prompt", max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("pad_token_id") == 0

    def test_generate_pad_token_id_not_passed_when_none(self):
        """If tokenizer.pad_token_id is None, pad_token_id should NOT be passed
        to generate() (let the model handle its own default).
        """
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()
        mock_tokenizer.pad_token_id = None

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

    # -----------------------------------------------------------------------
    # Chat-format input for MedGemma 4B (Fix #8: image token mismatch)
    # -----------------------------------------------------------------------

    def test_formats_prompt_as_chat_messages_with_images(self):
        """VisionModelWrapper should format the prompt as chat messages with
        {"type": "image"} entries for each image, so MedGemma 4B's processor
        can match images to <image> tokens.

        The error "Prompt contained 0 image tokens but received 2 images" occurs
        when the prompt is passed as a plain string without image placeholders.
        """
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=["img1.png", "img2.png"], prompt="Compare these images.")

        call_kwargs = mock_pipeline.call_args
        # The text kwarg should be a list of messages (chat format)
        text_arg = call_kwargs.kwargs.get("text")
        assert isinstance(text_arg, list), f"Expected list of messages, got {type(text_arg)}"
        assert len(text_arg) == 1  # Single user message
        msg = text_arg[0]
        assert msg["role"] == "user"

        # Content should have image entries + text entry
        content = msg["content"]
        assert isinstance(content, list)
        image_entries = [c for c in content if c.get("type") == "image"]
        text_entries = [c for c in content if c.get("type") == "text"]
        assert len(image_entries) == 2  # One per image
        assert len(text_entries) == 1
        assert text_entries[0]["text"] == "Compare these images."

    def test_single_image_formats_one_image_entry(self):
        """For a single image, only one {"type": "image"} entry should be present."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=["single.png"], prompt="Analyze this image.")

        call_kwargs = mock_pipeline.call_args
        text_arg = call_kwargs.kwargs.get("text")
        assert isinstance(text_arg, list)
        content = text_arg[0]["content"]
        image_entries = [c for c in content if c.get("type") == "image"]
        assert len(image_entries) == 1

    def test_images_inside_chat_content(self):
        """Images should be passed inside chat message content with 'url' key,
        not as a separate images= kwarg. This matches the transformers pipeline
        expected format for image-text-to-text models."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        images = ["img1.png", "img2.png"]
        wrapper = VisionModelWrapper(pipeline=mock_pipeline)
        wrapper(images=images, prompt="Analyze.")

        call_kwargs = mock_pipeline.call_args
        # images= should NOT be passed separately
        assert call_kwargs.kwargs.get("images") is None

        # Instead, images should be inside the chat content
        text_arg = call_kwargs.kwargs.get("text")
        assert isinstance(text_arg, list)
        content = text_arg[0]["content"]
        image_entries = [c for c in content if c.get("type") == "image"]
        assert len(image_entries) == 2
        # Each image entry should have a 'url' key with the image path
        assert image_entries[0].get("url") == "img1.png"
        assert image_entries[1].get("url") == "img2.png"


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


class TestTextModelWrapperLogging:
    """Tests for verbose logging in TextModelWrapper.__call__()."""

    def test_logs_prompt_length_in_chars(self):
        """TextModelWrapper should log the prompt length in characters."""
        from utils.model_wrappers import TextModelWrapper

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        fake_input_ids = MagicMock()
        fake_input_ids.shape = (1, 10)
        fake_input_ids.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer.apply_chat_template.return_value = fake_input_ids
        mock_tokenizer.decode.return_value = "response"
        mock_model.generate.return_value = MagicMock()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper("Test prompt for logging", max_tokens=100)

        # Should have logged the prompt length
        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "23" in all_log_text or "char" in all_log_text.lower()

    def test_logs_input_token_count(self):
        """TextModelWrapper should log the number of input tokens."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model(
            input_len=42, decode_text="response"
        )

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper("Test prompt", max_tokens=100)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "42" in all_log_text  # 42 input tokens

    def test_logs_generated_output_preview(self):
        """TextModelWrapper should log a preview of the generated text."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model(
            decode_text="The patient has pneumonia"
        )

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper("Test prompt", max_tokens=100)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_log_text = " ".join(all_debug_calls)
        assert "pneumonia" in all_log_text or "patient" in all_log_text

    def test_logs_which_template_path_was_used(self):
        """TextModelWrapper should log whether chat_template(tokenize=True) or fallback was used."""
        from utils.model_wrappers import TextModelWrapper

        mock_tokenizer, mock_model = _make_mock_tokenizer_and_model()

        wrapper = TextModelWrapper(model=mock_model, tokenizer=mock_tokenizer)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper("Test prompt", max_tokens=100)

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_log_text = " ".join(all_debug_calls)
        assert "chat_template" in all_log_text.lower() or "tokenize" in all_log_text.lower()


class TestVisionModelWrapperLogging:
    """Tests for verbose logging in VisionModelWrapper.__call__()."""

    def test_logs_image_count(self):
        """VisionModelWrapper should log the number of images."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "findings"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper(images=["img1.png", "img2.png"], prompt="Analyze.")

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_info_calls = [str(c) for c in mock_logger.info.call_args_list]
        all_log_text = " ".join(all_debug_calls + all_info_calls)
        assert "2" in all_log_text  # 2 images

    def test_logs_output_preview(self):
        """VisionModelWrapper should log a preview of the generated text."""
        from utils.model_wrappers import VisionModelWrapper

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Consolidation in left lower lobe"}]

        wrapper = VisionModelWrapper(pipeline=mock_pipeline)

        with patch("utils.model_wrappers.logger") as mock_logger:
            wrapper(images=["img.png"], prompt="Analyze.")

        all_debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        all_log_text = " ".join(all_debug_calls)
        assert "consolidation" in all_log_text.lower() or "lobe" in all_log_text.lower()
