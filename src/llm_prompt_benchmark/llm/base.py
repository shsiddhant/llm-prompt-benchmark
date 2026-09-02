from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class GeneratedResponse:
    text: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency: float
    finish_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> GeneratedResponse: ...
