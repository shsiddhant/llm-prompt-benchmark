from dataclasses import dataclass
from typing import Self

from llm_prompt_benchmark.benchmark import BenchmarkResult
from llm_prompt_benchmark.evaluation import StrategyEvaluation
from llm_prompt_benchmark.prompts import PromptStrategy


@dataclass(frozen=True)
class StrategyReport:
    strategy: PromptStrategy
    evaluation: StrategyEvaluation
    average_latency: float
    average_input_tokens: float
    average_output_tokens: float


@dataclass(frozen=True)
class BenchmarkReport:
    model: str
    reports: list[StrategyReport]

    @classmethod
    def generate(
        cls,
        results: list[BenchmarkResult],
        evaluations: list[StrategyEvaluation],
        model_name: str,
    ) -> Self:
        """
        Generate benchmark report from list of benchmark results and
        strategy evaluations.
        """
        reports: list[StrategyReport] = []
        for eval in evaluations:
            strategy_results = [
                result for result in results if result.strategy == eval.strategy
            ]
            average_latency = (
                sum(result.response.latency for result in strategy_results)
                / len(strategy_results)
                if strategy_results
                else 0.0
            )
            average_input_tokens = (
                sum(result.response.input_tokens or 0 for result in strategy_results)
                / len(strategy_results)
                if strategy_results
                else 0.0
            )
            average_output_tokens = (
                sum(result.response.output_tokens or 0 for result in strategy_results)
                / len(strategy_results)
                if strategy_results
                else 0.0
            )
            reports.append(
                StrategyReport(
                    strategy=eval.strategy,
                    evaluation=eval,
                    average_latency=average_latency,
                    average_input_tokens=average_input_tokens,
                    average_output_tokens=average_output_tokens,
                )
            )
        return cls(model=model_name, reports=reports)
