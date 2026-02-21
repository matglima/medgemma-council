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
                # We got token IDs directly from apply_chat_template.
                # Move to model device and build attention_mask.
                if self.device == "auto":
                    if hasattr(self.model, "device"):
                        input_ids = input_ids.to(self.model.device)
                else:
                    input_ids = input_ids.to(self.device)

                # Create attention_mask of all 1s (no padding in this path)
                try:
                    import torch
                    attention_mask = torch.ones_like(input_ids)
                except ImportError:
                    attention_mask = None

            input_len = input_ids.shape[1]

            # 2. Generate (greedy decoding by default to avoid sampling-mode
            #    amplification of numerical errors from 4-bit dequantization)
            generate_kwargs = {"do_sample": False}
            # Pass pad_token_id explicitly to suppress the
            # "Setting pad_token_id to eos_token_id" warning.
            if self.tokenizer.pad_token_id is not None:
                generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            generate_kwargs.update(kwargs)  # allow callers to override

            gen_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                gen_inputs["attention_mask"] = attention_mask

            output_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_tokens,
                **generate_kwargs,
            )

            # 3. Strip input tokens from output (only keep generated)
            generated_ids = output_ids[0][input_len:]

            # 4. Decode generated tokens
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
            result = self.pipeline(
                images=images,
                text=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

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
