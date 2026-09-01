import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from llm_prompt_benchmark.llm.base import GeneratedResponse

from .base import LLM


class HuggingFaceClient(LLM):
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, model: str, max_new_tokens: int = 16) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str) -> GeneratedResponse:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        target_device = next(self.model.parameters()).device

        inputs = inputs.to(target_device)

        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs, max_new_tokens=self.max_new_tokens
            )
        latency = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0, input_len:]

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()  # pyright: ignore[reportAttributeAccessIssue]
        return GeneratedResponse(
            text=text,
            input_tokens=input_len,
            output_tokens=len(generated_tokens),
            reasoning_tokens=None,
            latency=latency,
            finish_reason=None,
        )
