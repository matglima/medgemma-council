"""
Model inference wrappers for MedGemma-Council.

Bridges the gap between raw transformers models and the callable interface
expected by council agents:

- TextModelWrapper: wraps AutoModelForCausalLM + AutoTokenizer
    -> callable(prompt, max_tokens=N) -> {"choices": [{"text": "..."}]}

- VisionModelWrapper: wraps transformers pipeline("image-text-to-text")
    -> callable(images=..., prompt=...) -> [{"generated_text": "..."}]

- MockModelWrapper: deterministic mock for testing/development
    -> supports both text and vision output formats

Agent interface contract:
    Text agents:  result = self.llm(prompt, max_tokens=1024)
                  if isinstance(result, dict): text = result["choices"][0]["text"]
                  else: text = str(result)

    Vision agent: result = self.llm(images=images, prompt=prompt)
                  if isinstance(result, list): text = result[0]["generated_text"]
                  if isinstance(result, dict): text = result["generated_text"]
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import torch at module level for easier mocking in tests.
# In environments without torch, this will be None and functions will
# gracefully fall back to defaults.
try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore


class TextModelWrapper:
    """
    Wraps a transformers AutoModelForCausalLM + AutoTokenizer into a
    callable that matches the llama-cpp-python interface expected by agents.

    Usage:
        wrapper = TextModelWrapper(model=model, tokenizer=tokenizer)
        result = wrapper("What is the diagnosis?", max_tokens=1024)
        # result == {"choices": [{"text": "The patient presents with..."}]}
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "auto",
        max_input_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_tokens = max_input_tokens

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Tokenize prompt, run model.generate(), decode, and return
        in llama-cpp format: {"choices": [{"text": "..."}]}.

        Args:
            prompt: The text prompt to send to the model.
            max_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional kwargs passed to model.generate().

        Returns:
            Dict in llama-cpp format with generated text.
        """
        try:
            logger.debug(
                f"TextModelWrapper: prompt={len(prompt)} chars, "
                f"max_tokens={max_tokens}, max_input_tokens={self.max_input_tokens}"
            )

            # 1a. Apply chat template with tokenization in a single step.
            #     Using tokenize=True with truncation lets the chat template engine
            #     handle truncation *while preserving structural markers* like
            #     <end_of_turn> and <start_of_turn>model.  The old approach
            #     (tokenize=False → string → tokenizer(truncation=True)) truncated
            #     from the right, cutting off the model-turn markers that instruct
            #     the model to begin generation → empty specialist outputs.
            messages = [{"role": "user", "content": prompt}]
            input_ids = None
            used_chat_template_tokenize = False

            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                used_chat_template_tokenize = True
            except Exception:
                # Fallback: tokenizer doesn't support truncation/return_tensors
                # in apply_chat_template.  Try string-based approach.
                pass

            if not used_chat_template_tokenize:
                # Fallback path A: apply_chat_template(tokenize=False) -> string
                formatted_prompt = prompt
                try:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    # Tokenizer lacks chat template support entirely — use raw prompt
                    formatted_prompt = prompt

                # Tokenize the string (with truncation, may lose structural markers)
                if self.device == "auto":
                    tok_out = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens,
                    )
                    if hasattr(self.model, "device"):
                        tok_out = {
                            k: v.to(self.model.device) for k, v in tok_out.items()
                        }
                else:
                    tok_out = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens,
                    )
                    tok_out = {k: v.to(self.device) for k, v in tok_out.items()}

                input_ids = tok_out["input_ids"]
                attention_mask = tok_out.get("attention_mask")
            else:
                # apply_chat_template(tokenize=True, return_tensors="pt") may return:
                # - A plain Tensor (just input_ids)
                # - A BatchEncoding dict-like with 'input_ids' and 'attention_mask' keys
                # Handle both cases. Check for "input_ids" key presence specifically.
                # Use try/except for robustness across different object types.
                try:
                    # If input_ids has an "input_ids" key, it's a BatchEncoding
                    if "input_ids" in input_ids:
                        attention_mask = input_ids.get("attention_mask")
                        input_ids = input_ids["input_ids"]
                    else:
                        attention_mask = None
                except (TypeError, KeyError):
                    # Not a dict-like object, must be a plain tensor
                    attention_mask = None

                # Move to model device
                if self.device == "auto":
                    if hasattr(self.model, "device"):
                        input_ids = input_ids.to(self.model.device)
                else:
                    input_ids = input_ids.to(self.device)

                # Create attention_mask if not provided
                if attention_mask is None:
                    if torch is not None:
                        attention_mask = torch.ones_like(input_ids)
                    else:
                        attention_mask = None
                else:
                    attention_mask = attention_mask.to(input_ids.device)

            input_len = input_ids.shape[1]

            logger.debug(
                f"TextModelWrapper: input_tokens={input_len}, "
                f"path={'chat_template(tokenize=True)' if used_chat_template_tokenize else 'fallback string tokenization'}"
            )

            # 2. Generate (greedy decoding by default to avoid sampling-mode
            #    amplification of numerical errors from 4-bit dequantization)
            generate_kwargs = {"do_sample": False}
            # Pass pad_token_id explicitly to suppress the
            # "Setting pad_token_id to eos_token_id" warning.
            if self.tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            # NOTE: Do NOT override eos_token_id here. The model's generation_config
            # may have multiple EOS tokens (e.g., [1, 106] for Gemma), and overriding
            # with a single ID breaks proper generation stopping.
            generate_kwargs.update(kwargs)  # allow callers to override

            # Debug: log tokenizer config
            logger.debug(
                f"TextModelWrapper: tokenizer config - "
                f"pad_token_id={self.tokenizer.pad_token_id}, "
                f"eos_token_id={self.tokenizer.eos_token_id}, "
                f"bos_token_id={self.tokenizer.bos_token_id}"
            )
            
            # Debug: check model's generation config
            if hasattr(self.model, 'generation_config'):
                gc = self.model.generation_config
                logger.debug(
                    f"TextModelWrapper: model.generation_config - "
                    f"eos_token_id={getattr(gc, 'eos_token_id', 'N/A')}, "
                    f"pad_token_id={getattr(gc, 'pad_token_id', 'N/A')}, "
                    f"bos_token_id={getattr(gc, 'bos_token_id', 'N/A')}"
                )

            gen_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                gen_inputs["attention_mask"] = attention_mask

            # Debug: Run a single forward pass to check logits validity
            # This helps diagnose quantization issues that produce all-zero outputs
            try:
                with torch.no_grad() if torch is not None else None:
                    # Move input to model device for forward pass
                    test_input = input_ids
                    if hasattr(self.model, 'device'):
                        test_input = test_input.to(self.model.device)
                    # Get logits for the last input token position
                    outputs = self.model(test_input, use_cache=False)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits[0, -1, :]
                        # Check for NaN/Inf in logits
                        if torch is not None:
                            has_nan = torch.isnan(logits).any().item()
                            has_inf = torch.isinf(logits).any().item()
                            logits_min = logits.min().item()
                            logits_max = logits.max().item()
                            # Find the argmax token
                            predicted_token = logits.argmax().item()
                            logger.debug(
                                f"TextModelWrapper: logits check - "
                                f"has_nan={has_nan}, has_inf={has_inf}, "
                                f"range=[{logits_min:.2f}, {logits_max:.2f}], "
                                f"predicted_token={predicted_token}"
                            )
            except Exception as e:
                logger.debug(f"TextModelWrapper: could not check logits: {e}")

            output_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_tokens,
                **generate_kwargs,
            )

            # 3. Strip input tokens from output (only keep generated)
            # Debug: check raw output structure
            logger.debug(
                f"TextModelWrapper: raw output_ids type={type(output_ids)}, "
                f"shape={output_ids.shape if hasattr(output_ids, 'shape') else 'no shape'}"
            )
            # Log full output shape for debugging
            output_len = output_ids.shape[1] if hasattr(output_ids, 'shape') else len(output_ids[0])
            logger.debug(
                f"TextModelWrapper: input_len={input_len}, output_len={output_len}, "
                f"generated={output_len - input_len}"
            )
            
            # Debug: check what's in the full output before slicing
            try:
                import numpy as np
                full_seq = output_ids[0].cpu().numpy() if hasattr(output_ids[0], 'cpu') else np.array(output_ids[0])
                logger.debug(
                    f"TextModelWrapper: full_seq first_10={full_seq[:10].tolist()}, "
                    f"last_10={full_seq[-10:].tolist()}, "
                    f"at_input_len={full_seq[input_len:input_len+5].tolist() if input_len < len(full_seq) else 'N/A'}"
                )
            except Exception as e:
                logger.debug(f"TextModelWrapper: could not inspect full output: {e}")
            
            generated_ids = output_ids[0][input_len:]
            
            # Debug: log what we're about to decode
            if hasattr(generated_ids, 'shape'):
                gen_len = generated_ids.shape[0] if len(generated_ids.shape) > 0 else 1
                # Check for unique tokens
                try:
                    import numpy as np
                    unique_tokens = np.unique(generated_ids.cpu().numpy())
                    logger.debug(
                        f"TextModelWrapper: generated_ids len={gen_len}, "
                        f"unique_tokens={unique_tokens[:20].tolist()}, "
                        f"num_unique={len(unique_tokens)}"
                    )
                except Exception:
                    logger.debug(
                        f"TextModelWrapper: generated_ids len={gen_len}, "
                        f"first_5={list(generated_ids[:5].cpu().numpy()) if hasattr(generated_ids, 'cpu') else generated_ids[:5]}"
                    )
            else:
                gen_len = len(generated_ids) if hasattr(generated_ids, '__len__') else 1
                logger.debug(f"TextModelWrapper: generated_ids len={gen_len}")

            # 4. Decode generated tokens
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            logger.debug(
                f"TextModelWrapper: output_preview={text[:200]!r}"
            )

            return {"choices": [{"text": text}]}

        except Exception as e:
            logger.error(f"TextModelWrapper inference error: {e}")
            return {"choices": [{"text": f"Error during inference: {e}"}]}


class VisionModelWrapper:
    """
    Wraps a transformers pipeline("image-text-to-text") into a callable
    that matches the interface expected by RadiologyAgent.

    Usage:
        wrapper = VisionModelWrapper(pipeline=pipe)
        result = wrapper(images=["chest_xray.png"], prompt="Analyze this image.")
        # result == [{"generated_text": "The chest X-ray shows..."}]
    """

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def __call__(
        self,
        images: Optional[List[Any]] = None,
        prompt: str = "",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        Run the vision pipeline on images with a text prompt.

        Args:
            images: List of image paths or PIL Image objects.
            prompt: Text prompt/instruction for image analysis.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional kwargs passed to the pipeline.

        Returns:
            List of dicts: [{"generated_text": "..."}]
        """
        if not images:
            return [{"generated_text": "Error: No images provided for analysis."}]

        try:
            logger.debug(
                f"VisionModelWrapper: images={len(images)}, "
                f"prompt={len(prompt)} chars"
            )

            # Format the prompt as chat messages with image entries.
            # MedGemma 4B IT's pipeline expects images inside the message content:
            #   content = [
            #     {"type": "image", "url": image_path_or_pil},
            #     {"type": "text", "text": prompt}
            #   ]
            # The pipeline is called with just the messages, not images= separately.
            content = []
            for img in images:
                content.append({"type": "image", "url": img})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            result = self.pipeline(
                text=messages,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

            # Log output preview
            if isinstance(result, list) and result:
                preview = str(result[0].get("generated_text", ""))[:200]
            elif isinstance(result, dict):
                preview = str(result.get("generated_text", ""))[:200]
            else:
                preview = str(result)[:200]
            logger.debug(f"VisionModelWrapper: output_preview={preview!r}")

            # Normalize output to list format
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            else:
                return [{"generated_text": str(result)}]

        except Exception as e:
            logger.error(f"VisionModelWrapper inference error: {e}")
            return [{"generated_text": f"Error during vision inference: {e}"}]


class MockModelWrapper:
    """
    Deterministic mock wrapper for testing and development.

    Returns consistent, well-formatted outputs matching the expected
    interface for either text or vision mode.

    Usage:
        # Text mode (default):
        mock_llm = MockModelWrapper(mode="text")
        result = mock_llm("prompt", max_tokens=100)
        # -> {"choices": [{"text": "Mock clinical response."}]}

        # Vision mode:
        mock_llm = MockModelWrapper(mode="vision")
        result = mock_llm(images=["img.png"], prompt="Analyze.")
        # -> [{"generated_text": "Mock radiology findings."}]
    """

    DEFAULT_TEXT_RESPONSE = (
        "Based on the clinical presentation, the differential diagnosis includes "
        "the conditions discussed. Recommend further workup as outlined. "
        "Cite: AHA/ACC Guidelines 2023."
    )
    DEFAULT_VISION_RESPONSE = (
        "Radiological assessment: The imaging study demonstrates findings consistent "
        "with the clinical history. No acute abnormalities identified. "
        "Recommend clinical correlation."
    )

    def __init__(
        self,
        mode: str = "text",
        response: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.response = response

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Return mock output in the appropriate format."""
        if self.mode == "vision":
            text = self.response or self.DEFAULT_VISION_RESPONSE
            return [{"generated_text": text}]
        else:
            text = self.response or self.DEFAULT_TEXT_RESPONSE
            return {"choices": [{"text": text}]}
