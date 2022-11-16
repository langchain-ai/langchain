from formatter import DEFAULT_FORMATTER_MAPPING, SimpleFormatter

from abc import ABC
from langchain.datasets import Example
from typing import Any, Dict, List


class BasePrompt(ABC):
    """Base class for prompts."""

    pass


class FewShotPrompt(BasePrompt):
    """A prompt that is used for few-shot learning."""

    def __init__(
        self,
        example_template: str,
        examples: List[Example] = [],
        prompt_formatter="f-string",
        header: str = "",
        footer: str = "",
    ):
        self.examples = examples
        if prompt_formatter not in DEFAULT_FORMATTER_MAPPING:
            valid_formats = list(DEFAULT_FORMATTER_MAPPING.keys())
            raise ValueError(
                f"Invalid template format. Got `{prompt_formatter}`;"
                f" should be one of {valid_formats}"
            )
        if self.prompt_formatter == "simple":
            self.prompt_formatter = SimpleFormatter()
        else:
            self.prompt_formatter = DEFAULT_FORMATTER_MAPPING[prompt_formatter](
                example_template
            )
        self.header = header
        self.footer = footer

    def add_example(self, example: Example):
        self.examples.append(example)

    def __call__(self) -> str:
        body = "".join(
            [self.prompt_formatter.apply(example) for example in self.examples]
        )
        return f"{self.header}{body}{self.footer}"

    def __str__(self) -> str:
        return self()

    def __len__(self) -> int:
        return len(self.examples)


class ZeroShotPrompt(FewShotPrompt):
    """A prompt that is used for zero-shot learning.
    """

    def __init__(
        self,
        example_template: str,
        prompt_formatter="f-string",
        header: str = "",
        footer: str = "",
    ):
        super().__init__(example_template, [], prompt_formatter, header, footer)

    def add_example(self, example: Example):
        raise NotImplementedError("Zero-shot prompts do not support adding examples.")
