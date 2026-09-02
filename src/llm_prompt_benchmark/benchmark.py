from dataclasses import dataclass

from datasets import Dataset

from .llm.base import LLM, GeneratedResponse
from .prompts import PromptStrategy, build_prompt


@dataclass
class BenchmarkResult:
    strategy: PromptStrategy
    expected_label: int
    response: GeneratedResponse

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "expected_label": self.expected_label,
            "response": self.response.to_dict(),
        }


def run_benchmark(dataset: Dataset, llm: LLM) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for d in dataset:
        for strategy in PromptStrategy:
            prompt = build_prompt(d["text"], strategy)  # pyright: ignore[reportArgumentType, reportCallIssue]
            response = llm.generate(prompt)
            results.append(
                BenchmarkResult(
                    strategy=strategy,
                    expected_label=d["label"],  # pyright: ignore[reportArgumentType, reportCallIssue]
                    response=response,
                )
            )
    return results
