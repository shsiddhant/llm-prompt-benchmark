import pytest

from llm_prompt_benchmark.evaluation import ConfusionMatrix, Prediction, Sentiment
from llm_prompt_benchmark.prompts import PromptStrategy


def test_confusion_matrix_from_valid_predictions():

    predictions = [
        Prediction(
            strategy=PromptStrategy.ZERO_SHOT,
            expected=1,
            predicted=Sentiment.POSITIVE,
        ),
        Prediction(
            strategy=PromptStrategy.ZERO_SHOT,
            expected=0,
            predicted=Sentiment.NEGATIVE,
        ),
        Prediction(
            strategy=PromptStrategy.ZERO_SHOT,
            expected=0,
            predicted=Sentiment.POSITIVE,
        ),
        Prediction(
            strategy=PromptStrategy.ZERO_SHOT,
            expected=1,
            predicted=Sentiment.NEGATIVE,
        ),
        Prediction(
            strategy=PromptStrategy.ZERO_SHOT,
            expected=1,
            predicted=None,
        ),
    ]

    cm = ConfusionMatrix.from_predictions(predictions)

    assert cm.true_positive == 1
    assert cm.false_positive == 1
    assert cm.false_negative == 1
    assert cm.true_negative == 1


def test_accuracy():
    cm = ConfusionMatrix(
        true_positive=40,
        false_positive=10,
        false_negative=20,
        true_negative=30,
    )

    assert cm.accuracy() == pytest.approx(0.7)


def test_precision():
    cm = ConfusionMatrix(
        true_positive=40,
        false_positive=10,
        false_negative=20,
        true_negative=30,
    )

    assert cm.precision() == pytest.approx(0.8)


def test_recall():
    cm = ConfusionMatrix(
        true_positive=40,
        false_positive=10,
        false_negative=20,
        true_negative=30,
    )

    assert cm.recall() == pytest.approx(2 / 3)


def test_f1():
    cm = ConfusionMatrix(
        true_positive=40,
        false_positive=10,
        false_negative=20,
        true_negative=30,
    )

    assert cm.f_one() == pytest.approx(8 / 11)


def test_zero_division():
    cm = ConfusionMatrix(
        true_positive=0,
        true_negative=0,
        false_positive=0,
        false_negative=0,
    )

    assert cm.accuracy() == 0.0
    assert cm.precision() == 0.0
    assert cm.recall() == 0.0
    assert cm.f_one() == 0.0


def test_no_positive_predictions():
    cm = ConfusionMatrix(
        true_positive=0,
        true_negative=80,
        false_positive=0,
        false_negative=20,
    )

    assert cm.accuracy() == pytest.approx(0.8)
    assert cm.precision() == 0.0
    assert cm.recall() == 0.0
    assert cm.f_one() == 0.0
