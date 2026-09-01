from enum import Enum

ZERO_SHOT_TEMPLATE = """
Classify the sentiment of the following movie review as positive or negative.

Review:
{review}

Respond with exactly one word: positive or negative
"""

FEW_SHOT_TEMPLATE = """
Classify the sentiment of the movie review.

Examples:

Review:
A wonderful film with brilliant performances and a touching story.
Sentiment: positive

Review:
The plot was boring and the acting was terrible.
Sentiment: negative

Review:
{review}
Sentiment:

Respond with exactly one word: positive or negative
"""

INSTRUCTION_TEMPLATE = """
You are an expert sentiment analysis classifier.

Determine whether the movie review expresses an overall positive or negative
sentiment.

Classification rules:
- positive: the reviewer expresses an overall favorable opinion.
- negative: the reviewer expresses an overall unfavorable opinion.
- Base the classification on the review as a whole, not isolated words.
- Do not provide an explanation.

Review:
{review}

Respond with exactly one word: positive or negative
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
