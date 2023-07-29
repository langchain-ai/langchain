# flake8: noqa
# Credit to https://github.com/openai/evals/tree/main

from typing import Literal, Union
from langchain.prompts import PromptTemplate


BINARY_STRATEGY = """Does the submission meet the criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter "Y" or "N" again by itself on a new line."""
# Score on a scale from 1 to 10
SCORING_STRATEGY = """How well does the submission meet the criteria? First, write out in a step by step manner your\
 reasoning about each criterion to ensure that your conclusion is accurate. Avoid simply stating the scores at the outset.\
 After evaluating each criterion, assign a score from 0 to 9 where 0 means the submission does not meet the\
 criteria at all or the criteria does not describe the submission at all and 9 means the submission fully meets\
 the criteria or the criteria perfectly describes the submission. Print the numeric score\
 (from 0 to 9, without quotes or punctuation) on its own line. At the end, repeat just the numeric score again by itself on a new line."""

CONFIDENCE_STRATEGY = """How confident are you that the submission meets the criteria? Think carefully about each\
criterion and your confidence that the submission complies with the criteria. "Extremely confident no" means you are\
certain the submission does not meet the criteria, or the criteria in no way describes the submission.
Reason by thinking step by step, then assign your confidence level using the following scale:

1. [[Extremely confident no]]
2. [[Very confident no]]
3. [[Slightly confident no]]
4. [[Somewhat confident no]]
5. [[Unsure]]
6. [[Somewhat confident yes]]
7. [[Slightly confident yes]]
8. [[Very confident yes]]
9. [[Extremely confident yes]]

Then print the corresponding confidence level in words on its own line. At the end, repeat the confidence level again, in words as it is shown above, by itself on a new line."""

_TEMPLATE = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: {input}
***
[Submission]: {output}
***
[Criteria]: {criteria}
***
[END DATA]
"""

PROMPT = PromptTemplate.from_template(_TEMPLATE + BINARY_STRATEGY)


_LABELED_TEMPLATE = """You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: {input}
***
[Submission]: {output}
***
[Criteria]: {criteria}
***
[Reference]: {reference}
***
[END DATA]
"""

PROMPT_WITH_REFERENCES = PromptTemplate.from_template(
    _LABELED_TEMPLATE + BINARY_STRATEGY
)

STRATEGY_TYPE = Union[Literal["binary"], Literal["score"], Literal["confidence"]]


def get_prompt_template(
    requires_references: bool,
    strategy: Union[
        Literal["binary"], Literal["score"], Literal["confidence"]
    ] = "binary",
) -> PromptTemplate:
    """Get the prompt template for the specified strategy and model type."""
    strat_map = {
        "binary": BINARY_STRATEGY,
        "score": SCORING_STRATEGY,
        "confidence": CONFIDENCE_STRATEGY,
    }
    if strategy not in strat_map:
        raise ValueError(
            f"Unrecognized evalution strategy {strategy}"
            f"\nMust be one of {list(strat_map.keys())}"
        )
    template = _LABELED_TEMPLATE if requires_references else _TEMPLATE
    suffix = strat_map[strategy]
    return PromptTemplate.from_template(template + suffix + "\nReasoning:")
