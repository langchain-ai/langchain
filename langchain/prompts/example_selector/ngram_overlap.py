"""Select and order examples based on ngram overlap score (sentence_bleu score):
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
from typing import Dict, List, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pydantic import BaseModel, validator

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate

def ngram_overlap_score(source: List[str], example: List[str]) -> float:
    """Computes ngram overlap score (1<= n <=4) of source and example as sentence_bleu score:
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

    threshold: float = 0.0
    """Threshold score, at which algorithm stops."""

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select examples and ordering based on ngram overlap."""
        inputs = list(input_variables.values())
        examples = []
        print(f"inputs={inputs}")
        k = len(self.examples)
        score = [0] * k
        for i in range(k):
            score[i] = ngram_overlap_score(
                inputs, [self.examples[i][self.example_prompt.input_variables[0]]]
            )
            print(type([self.examples[i][self.example_prompt.input_variables[0]]]))
            print([self.examples[i][self.example_prompt.input_variables[0]]])
            print(score[i])

        while True:
            arg_max = np.argmax(score)
            if score[arg_max] <= self.threshold:
                break

            examples.append(self.examples[arg_max])
            print(examples)
            score[arg_max] = -1

        return examples
