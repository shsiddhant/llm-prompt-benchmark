from dataclasses import dataclass
from enum import IntEnum

from llm_prompt_benchmark.benchmark import BenchmarkResult
from llm_prompt_benchmark.prompts import PromptStrategy


class Sentiment(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1


@dataclass(frozen=True)
class Prediction:
    strategy: PromptStrategy
    predicted: Sentiment | None
    expected: int

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "predicted": self.predicted,
            "expected": self.expected,
        }


def parse_benchmark_result(result: BenchmarkResult) -> Prediction:
    text = result.response.text

    normalized_text = text.strip().lower() if text else None
    predicted = None

    if normalized_text == "positive":
        predicted = Sentiment.POSITIVE
    if normalized_text == "negative":
        predicted = Sentiment.NEGATIVE
    return Prediction(
        strategy=result.strategy, predicted=predicted, expected=result.expected_label
    )


def predict(results: list[BenchmarkResult]) -> list[Prediction]:
    return [parse_benchmark_result(result) for result in results]
