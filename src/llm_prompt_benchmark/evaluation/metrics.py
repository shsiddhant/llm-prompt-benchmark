from dataclasses import asdict, dataclass
from typing import Self

from .predict import Prediction


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def to_dict(self) -> dict:
        return asdict(self)

    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    def accuracy(self) -> float:
        deno = self.total()
        return (self.true_positive + self.true_negative) / deno if deno else 0.0

    def precision(self) -> float:
        deno = self.true_positive + self.false_positive
        return self.true_positive / deno if deno else 0.0

    def recall(self) -> float:
        deno = self.true_positive + self.false_negative
        return self.true_positive / deno if deno else 0.0

    def f_one(self) -> float:
        deno = self.precision() + self.recall()
        return 2 * self.precision() * self.recall() / deno if deno else 0.0

    @classmethod
    def from_predictions(cls, predictions: list[Prediction]) -> Self:
        """
        Construct confusion matrix from list of valid predictions.

        A prediction is valid if its predicted value is not None

        Parameters
        ----------
        predictions: list[Prediction]
            A list of valid predictions.

        Returns
        -------
        ConfusionMatrix
            A confusion matrix.
        """

        tp, fp, fn, tn = 0, 0, 0, 0
        for p in predictions:
            if p.expected == 1 and p.predicted == 1:
                tp += 1
            elif p.expected == 0 and p.predicted == 1:
                fp += 1
            elif p.expected == 1 and p.predicted == 0:
                fn += 1
            elif p.expected == 0 and p.predicted == 0:
                tn += 1

        return cls(
            true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn
        )
