"""Select and order examples based on ngram overlap score (sentence_bleu score).

https://www.nltk.org/_modules/nltk/translate/bleu_score.html
https://aclanthology.org/P02-1040.pdf
"""
from typing import Dict, List

import numpy as np
from pydantic import BaseModel, root_validator

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate


def ngram_overlap_score(source: List[str], example: List[str]) -> float:
    """Compute ngram overlap score of source and example as sentence_bleu score.

    Use sentence_bleu with method1 smoothing function and auto reweighting.
    Return float value between 0.0 and 1.0 inclusive.
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
    from nltk.translate.bleu_score import (  # type: ignore
        SmoothingFunction,
        sentence_bleu,
    )

    hypotheses = source[0].split()
    references = [s.split() for s in example]

    return float(
        sentence_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1,
            auto_reweigh=True,
        )
    )


class NGramOverlapExampleSelector(BaseExampleSelector, BaseModel):
    """Select and order examples based on ngram overlap score (sentence_bleu score).

    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    threshold: float = -1.0
    """Threshold at which algorithm stops. Set to -1.0 by default.

    For negative threshold:
    select_examples sorts examples by ngram_overlap_score, but excludes none.
    For threshold greater than 1.0:
    select_examples excludes all examples, and returns an empty list.
    For threshold equal to 0.0:
    select_examples sorts examples by ngram_overlap_score,
    and excludes examples with no ngram overlap with input.
    """

    @root_validator(pre=True)
    def check_dependencies(cls, values: Dict) -> Dict:
        """Check that valid dependencies exist."""
        try:
            from nltk.translate.bleu_score import (  # noqa: disable=F401
                SmoothingFunction,
                sentence_bleu,
            )
        except ImportError as e:
            raise ValueError(
                "Not all the correct dependencies for this ExampleSelect exist"
            ) from e

        return values

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Return list of examples sorted by ngram_overlap_score with input.

        Descending order.
        Excludes any examples with ngram_overlap_score less than or equal to threshold.
        """
        inputs = list(input_variables.values())
        examples = []
        k = len(self.examples)
        score = [0.0] * k
        first_prompt_template_key = self.example_prompt.input_variables[0]

        for i in range(k):
            score[i] = ngram_overlap_score(
                inputs, [self.examples[i][first_prompt_template_key]]
            )

        while True:
            arg_max = np.argmax(score)
            if (score[arg_max] < self.threshold) or abs(
                score[arg_max] - self.threshold
            ) < 1e-9:
                break

            examples.append(self.examples[arg_max])
            score[arg_max] = self.threshold - 1.0

        return examples
