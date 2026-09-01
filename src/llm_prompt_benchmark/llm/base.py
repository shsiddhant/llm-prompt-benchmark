from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GeneratedResponse:
    text: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency: float
    finish_reason: str | None


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> GeneratedResponse: ...
