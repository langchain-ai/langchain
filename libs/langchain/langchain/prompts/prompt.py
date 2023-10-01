"""Prompt schema definition."""
from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional, Union

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    _get_jinja2_variables_from_template,
    check_valid_template,
)
from langchain.pydantic_v1 import root_validator


class PromptTemplate(StringPromptTemplate):
    """A prompt template for a language model.

    A prompt template consists of a string template. It accepts a set of parameters
    from the user that can be used to generate a prompt for a language model.

    The template can be formatted using either f-strings (default) or jinja2 syntax.

    Example:

        .. code-block:: python

            from langchain.prompts import PromptTemplate

            # Instantiation using from_template (recommended)
            prompt = PromptTemplate.from_template("Say {foo}")
            prompt.format(foo="bar")

            # Instantiation using initializer
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        return {
            "template_format": self.template_format,
        }

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = True
    """Whether or not to try validating the template."""

    def __add__(self, other: Any) -> PromptTemplate:
        """Override the + operator to allow for combining prompt templates."""
        # Allow for easy combining
        if isinstance(other, PromptTemplate):
            if self.template_format != "f-string":
                raise ValueError(
                    "Adding prompt templates only supported for f-strings."
                )
            if other.template_format != "f-string":
                raise ValueError(
                    "Adding prompt templates only supported for f-strings."
                )
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )
            template = self.template + other.template
            # If any do not want to validate, then don't
            validate_template = self.validate_template and other.validate_template
            partial_variables = {k: v for k, v in self.partial_variables.items()}
            for k, v in other.partial_variables.items():
                if k in partial_variables:
                    raise ValueError("Cannot have same variable partialed twice.")
                else:
                    partial_variables[k] = v
            return PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format="f-string",
                validate_template=validate_template,
            )
        elif isinstance(other, str):
            prompt = PromptTemplate.from_template(other)
            return self + prompt
        else:
            raise NotImplementedError(f"Unsupported operand type for +: {type(other)}")

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

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
    def from_template(
        cls,
        template: str,
        *,
        template_format: str = "f-string",
        partial_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a prompt template from a template.

        Args:
            template: The template to load.
            template_format: The format of the template. Use `jinja2` for jinja2,
                             and `f-string` or None for f-strings.
            partial_variables: A dictionary of variables that can be used to partially
                               fill in the template. For example, if the template is
                              `"{variable1} {variable2}"`, and `partial_variables` is
                              `{"variable1": "foo"}`, then the final prompt will be
                              `"foo {variable2}"`.

        Returns:
            The prompt template loaded from the template.
        """
        if template_format == "jinja2":
            # Get the variables for the template
            input_variables = _get_jinja2_variables_from_template(template)
        elif template_format == "f-string":
            input_variables = {
                v for _, v, _, _ in Formatter().parse(template) if v is not None
            }
        else:
            raise ValueError(f"Unsupported template format: {template_format}")

        _partial_variables = partial_variables or {}

        if _partial_variables:
            input_variables = {
                var for var in input_variables if var not in _partial_variables
            }

        return cls(
            input_variables=sorted(input_variables),
            template=template,
            template_format=template_format,
            partial_variables=_partial_variables,
            **kwargs,
        )


# For backwards compatibility.
Prompt = PromptTemplate
