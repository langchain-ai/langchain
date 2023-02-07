"""Prompt schema definition."""
from __future__ import annotations

from string import Formatter
from typing import Any, Dict, List

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    check_valid_template,
)


class PromptTemplate(BasePromptTemplate, BaseModel):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = True
    """Whether or not to try validating the template."""

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

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
        if values["validate_template"]:
            check_valid_template(
                values["template"], values["template_format"], values["input_variables"]
            )
        return values

    @classmethod
    def from_examples(
        cls,
        examples: List[str],
        suffix: str,
        input_variables: List[str],
        example_separator: str = "\n\n",
        prefix: str = "",
    ) -> PromptTemplate:
        """Take examples in list format with prefix and suffix to create a prompt.

        Intended be used as a way to dynamically create a prompt from examples.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            example_separator: The separator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.

        Returns:
            The final prompt generated.
        """
        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_file(
        cls, template_file: str, input_variables: List[str]
    ) -> PromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            input_variables: A list of variable names the final prompt template
                will expect.
        Returns:
            The prompt loaded from the file.
        """
        with open(template_file, "r") as f:
            template = f.read()
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        """Load a prompt template from a template."""
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
        return cls(input_variables=list(sorted(input_variables)), template=template)


# For backwards compatibility.
Prompt = PromptTemplate
