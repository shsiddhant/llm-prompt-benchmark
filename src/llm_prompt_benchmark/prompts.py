from enum import Enum

ZERO_SHOT_TEMPLATE = """
Classify the sentiment of the following movie review as positive or negative.

Review:
{review}

Respond with exactly one word:
positive
or
negative
"""

FEW_SHOT_TEMPLATE = """
Classify the sentiment of the movie review.

Review:
A wonderful film with brilliant performances and a touching story.
Sentiment: positive

Review:
The plot was boring and the acting was terrible.
Sentiment: negative

Review:
{review}
Sentiment:
"""

INSTRUCTION_TEMPLATE = """
You are an expert sentiment analysis classifier.

Read the movie review carefully.
Determine whether the overall sentiment is positive or negative.
Return exactly one word: positive or negative.

Review:
{review}
"""


class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    INSTRUCTION = "instruction"

    @property
    def template(self):
        match self:
            case PromptStrategy.ZERO_SHOT:
                return ZERO_SHOT_TEMPLATE
            case PromptStrategy.FEW_SHOT:
                return FEW_SHOT_TEMPLATE
            case PromptStrategy.INSTRUCTION:
                return INSTRUCTION_TEMPLATE


def build_prompt(review: str, strategy: PromptStrategy) -> str:
    return strategy.template.strip().format(review=review)
