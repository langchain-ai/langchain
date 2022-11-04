"""Prompt schema definition."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.formatting import formatter

_FORMATTER_MAPPING = {
    "f-string": formatter.format,
}

def count_words(template: str) -> int:
    """Count words in a template."""
    all_lines = template.split("\n")
    all_words = []
    for line in all_lines:
        if line:
            all_words.extend(line.strip().split(" "))
    return len(all_words)


class BasePrompt(ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """


class Prompt(BaseModel, BasePrompt):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import Prompt
            prompt = Prompt(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string'."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
        return _FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        input_variables = values["input_variables"]
        template = values["template"]
        template_format = values["template_format"]
        if template_format not in _FORMATTER_MAPPING:
            valid_formats = list(_FORMATTER_MAPPING)
            raise ValueError(
                f"Invalid template format. Got `{template_format}`;"
                f" should be one of {valid_formats}"
            )
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        try:
            formatter_func = _FORMATTER_MAPPING[template_format]
            formatter_func(template, **dummy_inputs)
        except KeyError:
            raise ValueError("Invalid prompt schema.")
        return values

    @classmethod
    def from_examples(
        cls,
        examples: List[str],
        suffix: str,
        input_variables: List[str],
        example_separator: str = "\n\n",
        prefix: str = "",
    ) -> "Prompt":
        """Take examples in list format with prefix and suffix to create a prompt.

        Intended be used as a way to dynamically create a prompt from examples.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            example_separator: The seperator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.

        Returns:
            The final prompt generated.
        """
        example_str = example_separator.join(examples)
        template = prefix + example_str + suffix
        return cls(input_variables=input_variables, template=template)


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

    example_separator: str
    """Example separator, e.g. \n\n, for the dynamic prompt creation."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    prefix: str
    """Prefix for the prompt."""

    suffix: str
    """Suffix for the prompt."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string'."""

    get_text_length: Callable = count_words
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

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

        def return_template(example_list: List[str]) -> str:
            """Return template given example list."""
            template = self.example_separator.join(
                [self.prefix, *example_list, self.suffix]
            )
            return _FORMATTER_MAPPING[self.template_format](template, **kwargs)

        curr_examples = self.examples
        template = return_template(curr_examples)
        while self.get_text_length(template) > self.max_length and curr_examples:
            curr_examples = curr_examples[:-1]
            template = return_template(curr_examples)
        print(template)
        return template

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        input_variables = values["input_variables"]
        suffix = values["suffix"]
        template_format = values["template_format"]
        if template_format not in _FORMATTER_MAPPING:
            valid_formats = list(_FORMATTER_MAPPING)
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
        # TODO variables could be in prefix or suffix
        try:
            formatter_func = _FORMATTER_MAPPING[template_format]
            formatter_func(suffix, **dummy_inputs)
        except KeyError:
            raise ValueError("Invalid prompt schema.")
        return values
