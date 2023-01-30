"""Select and order examples based on ngram overlap score (sentence_bleu score):
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
from typing import Any, Dict, List, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pydantic import BaseModel, validator

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate

def ngram_overlap_score(source: List[str], example: list[str]) -> Any:
    """Computes ngram overlap score (1<= n <=4) of source and example as sentence_bleu score (0<= score <=1).
        Uses sentence_bleu with method1 smoothing function and auto reweighting.
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
    hypotheses = source[0].split()
    references = [s.split() for s in example]
    return sentence_bleu(
        references,
        hypotheses,
        smoothing_function=SmoothingFunction().method1,
        auto_reweigh=True,
    )

class NGramOverlapExampleSelector(BaseExampleSelector, BaseModel):
    """Select and order examples based on ngram overlap score (sentence_bleu score):
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    threshold: float = -1.0
    """Threshold at which algorithm stops. Set to -1.0 by default.
        If threshold is negative, select_examples will not exclude any example, only sort them by ngram_overlap_score.
        If threshold is greater than 1.0, select_examples will exclude all examples and return an empty list.
        If threshold is set to 0.0, select_examples will exclude examples with no ngram overlap with input.
    """

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Returns list of examples sorted, by ngram_overlap_score with input, in descending order.
            Excludes any examples whose ngram_overlap_score is less than or equal to threshold.
        """
        inputs = list(input_variables.values())
        examples = []
        print(f"inputs={inputs},type={type(inputs)}")
        k = len(self.examples)
        score = [0] * k
        first_prompt_template_key = self.example_prompt.input_variables[0]
        print(f"type={type(first_prompt_template_key)}")
        for i in range(k):
            score[i] = ngram_overlap_score(
                inputs, [self.examples[i][first_prompt_template_key]]           
            )
        print(score)

        while True:
            arg_max = np.argmax(score)
            if score[arg_max] <= self.threshold:
                break

            examples.append(self.examples[arg_max])
            print(examples)
            score[arg_max] = self.threshold - 1
            print(score)

        return examples
