"""Select random examples based on length."""
import random
import re
from typing import Callable, Dict, List

from langchain_core.example_selectors.length_based import LengthBasedExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel


def _get_length_based(text: str) -> int:
    return len(re.split("\n| ", text))


class RandomLengthExampleSelector(LengthBasedExampleSelector, BaseModel):
    """Select random examples based on length."""

    example_prompt: PromptTemplate
    min_remaining: int = 30
    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count."""

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select random examples to use based on the input lengths."""
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        indexes = list(range(0, len(self.examples)))
        random.shuffle(indexes)
        examples = []
        for i in indexes:
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                continue
            else:
                examples.append(self.examples[i])
                if new_length < self.min_remaining:
                    break
                remaining_length = new_length
        return examples
