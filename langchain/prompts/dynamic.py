"""Dynamic prompt schema definition."""
import re
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING, BasePrompt


class DynamicPrompt(BaseModel, BasePrompt):
    r"""Schema to represent a dynamic prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import DynamicPrompt
            dynamic_prompt = DynamicPrompt(
                examples=["Say hi. Hi", "Say ho. Ho"],
                example_separator="\n\n",
                prefix="",
                suffix="\n\nSay {foo}"
                input_variables=["foo"],
                max_length=200,
                get_text_length=word_count
            )
    """

    examples: List[str]
    """A list of the examples that the prompt template expects."""

    example_separator: str = "\n\n"
    """Example separator, e.g. \n\n, for the dynamic prompt creation."""

    input_variables: List[str] = []
    """A list of the names of the variables the prompt template expects."""

    prefix: str = ""
    """Prefix for the prompt."""

    suffix: str = ""
    """Suffix for the prompt."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string'."""

    get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def template(self, example_list: List[str], **kwargs: Any) -> str:
        """Return template given example list."""
        template = self.example_separator.join(
            [self.prefix, *example_list, self.suffix]
        )
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    def format(self, **kwargs: Any) -> str:
        """Dynamically format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        curr_examples = self.examples
        template = self.template(curr_examples, **kwargs)
        while self.get_text_length(template) > self.max_length and curr_examples:
            curr_examples = curr_examples[:-1]
            template = self.template(curr_examples, **kwargs)
        return template

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        input_variables = values["input_variables"]
        prefix = values["prefix"]
        suffix = values["suffix"]
        template_format = values["template_format"]
        if template_format not in DEFAULT_FORMATTER_MAPPING:
            valid_formats = list(DEFAULT_FORMATTER_MAPPING)
            raise ValueError(
                f"Invalid template format. Got `{template_format}`;"
                f" should be one of {valid_formats}"
            )
        try:
            result = values["get_text_length"]("foo")
            assert isinstance(result, int)
        except AssertionError:
            raise ValueError(
                "Invalid text length callable, must take string & return int;"
            )
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        try:
            formatter_func = DEFAULT_FORMATTER_MAPPING[template_format]
            formatter_func(prefix + suffix, **dummy_inputs)
        except KeyError:
            raise ValueError("Invalid prompt schema.")
        return values
