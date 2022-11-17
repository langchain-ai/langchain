from typing import Any, List, Callable
import re
from pydantic import BaseModel, validator
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector


class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    example_text_lengths: List[int]  #: :meta private:

    @validator("example_text_lengths", always=True)
    def calculate_example_text_lengths(cls, v, values):
        example_prompt = values["example_prompt"]
        get_text_length = values["get_text_length"]
        string_examples = [example_prompt.format(**eg) for eg in values["examples"]]
        return [get_text_length(eg) for eg in string_examples]

    def select_examples(self, **kwargs: Any) -> List[dict]:
        inputs = " ".join(kwargs.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if i < 0:
                break
            else:
                examples.append(self.examples[0])
                remaining_length = new_length
        return examples