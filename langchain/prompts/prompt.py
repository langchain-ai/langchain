"""Prompt schema definition."""
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING, BasePrompt


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
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        input_variables = values["input_variables"]
        template = values["template"]
        template_format = values["template_format"]
        if template_format not in DEFAULT_FORMATTER_MAPPING:
            valid_formats = list(DEFAULT_FORMATTER_MAPPING)
            raise ValueError(
                f"Invalid template format. Got `{template_format}`;"
                f" should be one of {valid_formats}"
            )
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        try:
            formatter_func = DEFAULT_FORMATTER_MAPPING[template_format]
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
