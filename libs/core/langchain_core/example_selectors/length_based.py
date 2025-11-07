"""Select examples based on length."""

import re
from collections.abc import Callable

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate


def _get_length_based(text: str) -> int:
    return len(re.split("\n| ", text))


class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """Select examples based on length."""

    examples: list[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    example_text_lengths: list[int] = Field(default_factory=list)
    """Length of each example."""

    def add_example(self, example: dict[str, str]) -> None:
        """Add new example to list.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        """
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    async def aadd_example(self, example: dict[str, str]) -> None:
        """Async add new example to list.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        """
        self.add_example(example)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Validate that the examples are formatted correctly."""
        if self.example_text_lengths:
            return self
        string_examples = [self.example_prompt.format(**eg) for eg in self.examples]
        self.example_text_lengths = [self.get_text_length(eg) for eg in string_examples]
        return self

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select which examples to use based on the input lengths.

        Args:
            input_variables: A dictionary with keys as input variables
               and values as their values.

        Returns:
            A list of examples to include in the prompt.
        """
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                break
            examples.append(self.examples[i])
            remaining_length = new_length
            i += 1
        return examples

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Async select which examples to use based on the input lengths.

        Args:
            input_variables: A dictionary with keys as input variables
               and values as their values.

        Returns:
            A list of examples to include in the prompt.
        """
        return self.select_examples(input_variables)
