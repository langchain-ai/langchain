"""Prompt schema definition."""
from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Union

from pydantic import Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    _get_jinja2_variables_from_template,
    check_valid_template,
)


class PromptTemplate(StringPromptTemplate):
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
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
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
        **kwargs: Any,
    ) -> PromptTemplate:
        """Take examples in list format with prefix and suffix to create a prompt.

        Intended to be used as a way to dynamically create a prompt from examples.

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
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_file(
        cls, template_file: Union[str, Path], input_variables: List[str], **kwargs: Any
    ) -> PromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            input_variables: A list of variable names the final prompt template
                will expect.
        Returns:
            The prompt loaded from the file.
        """
        with open(str(template_file), "r") as f:
            template = f.read()
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> PromptTemplate:
        """Load a prompt template from a template."""
        if "template_format" in kwargs and kwargs["template_format"] == "jinja2":
            # Get the variables for the template
            input_variables = _get_jinja2_variables_from_template(template)

        else:
            input_variables = {
                v for _, v, _, _ in Formatter().parse(template) if v is not None
            }

        if "partial_variables" in kwargs:
            partial_variables = kwargs["partial_variables"]
            input_variables = {
                var for var in input_variables if var not in partial_variables
            }

        return cls(
            input_variables=list(sorted(input_variables)), template=template, **kwargs
        )


# For backwards compatibility.
Prompt = PromptTemplate
