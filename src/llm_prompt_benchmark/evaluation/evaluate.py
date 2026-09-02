from dataclasses import dataclass

from llm_prompt_benchmark.evaluation.metrics import ConfusionMatrix
from llm_prompt_benchmark.evaluation.predict import Prediction
from llm_prompt_benchmark.prompts import PromptStrategy


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy: PromptStrategy
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float
    recall: float
    f_one: float
    valid_predictions: int
    invalid_predictions: int
    format_compliance: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f_one": self.f_one,
            "valid_predictions": self.valid_predictions,
            "invalid_predictions": self.invalid_predictions,
            "format_compliance": self.format_compliance,
        }


def evaluate(predictions: list[Prediction]) -> list[StrategyEvaluation]:
    evaluation: list[StrategyEvaluation] = []
    for strategy in PromptStrategy:
        strategy_predictions = [p for p in predictions if p.strategy == strategy]

        total = len(strategy_predictions)
        valid_predictions = sum(
            1 for p in strategy_predictions if p.predicted is not None
        )
        invalid_predictions = total - valid_predictions

        format_compliance = valid_predictions / total if total else 0.0

        confusion_matrix = ConfusionMatrix.from_predictions(strategy_predictions)

        evaluation.append(
            StrategyEvaluation(
                strategy=strategy,
                confusion_matrix=confusion_matrix,
                accuracy=confusion_matrix.accuracy(),
                precision=confusion_matrix.precision(),
                recall=confusion_matrix.recall(),
                f_one=confusion_matrix.recall(),
                valid_predictions=valid_predictions,
                invalid_predictions=invalid_predictions,
                format_compliance=format_compliance,
            )
        )
    return evaluation
