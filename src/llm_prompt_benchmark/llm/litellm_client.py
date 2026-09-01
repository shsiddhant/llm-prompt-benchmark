import time

from litellm import ModelResponse, Usage, completion

from .base import LLM, GeneratedResponse


class LLMClient(LLM):
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str) -> GeneratedResponse:
        start = time.perf_counter()
        response = completion(
            self.model, messages=[{"role": "user", "content": prompt}]
        )
        latency = time.perf_counter() - start

        if not isinstance(response, ModelResponse):
            raise TypeError("Expected `ModelResponse`.", response)

        choices = response.choices[0]
        message = choices.message
        usage: Usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
        completion_token_details = usage.completion_tokens_details
        reasoning_tokens = (
            completion_token_details.reasoning_tokens
            if completion_token_details
            else None
        )

        return GeneratedResponse(
            text=message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency=latency,
            finish_reason=choices.finish_reason,
        )
