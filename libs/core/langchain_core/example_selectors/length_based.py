"""Select examples based on length."""
import re
from dataclasses import field
from typing import Callable, Dict, List

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.load.serializable import Serializable
from langchain_core.prompts.prompt import PromptTemplate


def _get_length_based(text: str) -> int:
    return len(re.split("\n| ", text))


class LengthBasedExampleSelector(BaseExampleSelector, Serializable):
    """Select examples based on length."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    example_text_lengths: List[int] = field(default_factory=list)  #: :meta private:

    def __post_init__(self) -> None:
        # validate example_text_lengths
        # Check if text lengths were passed in
        if self.example_text_lengths:
            return
        else:
            # If they were not, calculate them
            example_prompt = self.example_prompt
            get_text_length = self.get_text_length
            string_examples = [example_prompt.format(**eg) for eg in self.examples]
            self.example_text_lengths = [get_text_length(eg) for eg in string_examples]

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    async def aadd_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.add_example(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                break
            else:
                examples.append(self.examples[i])
                remaining_length = new_length
            i += 1
        return examples

    async def aselect_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        return self.select_examples(input_variables)
