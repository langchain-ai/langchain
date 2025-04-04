"""Prompt schema definition."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import override

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
    mustache_schema,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class PromptTemplate(StringPromptTemplate):
    """Prompt template for a language model.

    A prompt template consists of a string template. It accepts a set of parameters
    from the user that can be used to generate a prompt for a language model.

    The template can be formatted using either f-strings (default), jinja2,
    or mustache syntax.

    *Security warning*:
        Prefer using `template_format="f-string"` instead of
        `template_format="jinja2"`, or make sure to NEVER accept jinja2 templates
        from untrusted sources as they may lead to arbitrary Python code execution.

        As of LangChain 0.0.329, Jinja2 templates will be rendered using
        Jinja2's SandboxedEnvironment by default. This sand-boxing should
        be treated as a best-effort approach rather than a guarantee of security,
        as it is an opt-out rather than opt-in approach.

        Despite the sand-boxing, we recommend to never use jinja2 templates
        from untrusted sources.

    Example:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate

            # Instantiation using from_template (recommended)
            prompt = PromptTemplate.from_template("Say {foo}")
            prompt.format(foo="bar")

            # Instantiation using initializer
            prompt = PromptTemplate(template="Say {foo}")
    """

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        return {
            "template_format": self.template_format,
        }

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "prompts", "prompt"]

    template: str
    """The prompt template."""

    template_format: PromptTemplateFormat = "f-string"
    """The format of the prompt template.
    Options are: 'f-string', 'mustache', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""

    @model_validator(mode="before")
    @classmethod
    def pre_init_validation(cls, values: dict) -> Any:
        """Check that template and input variables are consistent."""
        if values.get("template") is None:
            # Will let pydantic fail with a ValidationError if template
            # is not provided.
            return values

        # Set some default values based on the field defaults
        values.setdefault("template_format", "f-string")
        values.setdefault("partial_variables", {})

        if values.get("validate_template"):
            if values["template_format"] == "mustache":
                msg = "Mustache templates cannot be validated."
                raise ValueError(msg)

            if "input_variables" not in values:
                msg = "Input variables must be provided to validate the template."
                raise ValueError(msg)

            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )

        if values["template_format"]:
            values["input_variables"] = [
                var
                for var in get_template_variables(
                    values["template"], values["template_format"]
                )
                if var not in values["partial_variables"]
            ]

        return values

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """Get the input schema for the prompt.

        Args:
            config: The runnable configuration.

        Returns:
            The input schema for the prompt.
        """
        if self.template_format != "mustache":
            return super().get_input_schema(config)

        return mustache_schema(self.template)

    def __add__(self, other: Any) -> PromptTemplate:
        """Override the + operator to allow for combining prompt templates."""
        # Allow for easy combining
        if isinstance(other, PromptTemplate):
            if self.template_format != "f-string":
                msg = "Adding prompt templates only supported for f-strings."
                raise ValueError(msg)
            if other.template_format != "f-string":
                msg = "Adding prompt templates only supported for f-strings."
                raise ValueError(msg)
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )
            template = self.template + other.template
            # If any do not want to validate, then don't
            validate_template = self.validate_template and other.validate_template
            partial_variables = dict(self.partial_variables.items())
            for k, v in other.partial_variables.items():
                if k in partial_variables:
                    msg = "Cannot have same variable partialed twice."
                    raise ValueError(msg)
                partial_variables[k] = v
            return PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format="f-string",
                validate_template=validate_template,
            )
        if isinstance(other, str):
            prompt = PromptTemplate.from_template(other)
            return self + prompt
        msg = f"Unsupported operand type for +: {type(other)}"
        raise NotImplementedError(msg)

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
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @classmethod
    def from_examples(
        cls,
        examples: list[str],
        suffix: str,
        input_variables: list[str],
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
        cls,
        template_file: Union[str, Path],
        input_variables: Optional[list[str]] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            input_variables: [DEPRECATED] A list of variable names the final prompt
                template will expect. Defaults to None.
            encoding: The encoding system for opening the template file.
                If not provided, will use the OS default.

        input_variables is ignored as from_file now delegates to from_template().

        Returns:
            The prompt loaded from the file.
        """
        template = Path(template_file).read_text(encoding=encoding)
        if input_variables:
            warnings.warn(
                "`input_variables' is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls.from_template(template=template, **kwargs)

    @classmethod
    def from_template(
        cls,
        template: str,
        *,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a prompt template from a template.

        *Security warning*:
            Prefer using `template_format="f-string"` instead of
            `template_format="jinja2"`, or make sure to NEVER accept jinja2 templates
            from untrusted sources as they may lead to arbitrary Python code execution.

            As of LangChain 0.0.329, Jinja2 templates will be rendered using
            Jinja2's SandboxedEnvironment by default. This sand-boxing should
            be treated as a best-effort approach rather than a guarantee of security,
            as it is an opt-out rather than opt-in approach.

            Despite the sand-boxing, we recommend never using jinja2 templates
            from untrusted sources.

        Args:
            template: The template to load.
            template_format: The format of the template. Use `jinja2` for jinja2,
                             `mustache` for mustache, and `f-string` for f-strings.
                             Defaults to `f-string`.
            partial_variables: A dictionary of variables that can be used to partially
                               fill in the template. For example, if the template is
                              `"{variable1} {variable2}"`, and `partial_variables` is
                              `{"variable1": "foo"}`, then the final prompt will be
                              `"foo {variable2}"`. Defaults to None.
            kwargs: Any other arguments to pass to the prompt template.

        Returns:
            The prompt template loaded from the template.
        """
        input_variables = get_template_variables(template, template_format)
        _partial_variables = partial_variables or {}

        if _partial_variables:
            input_variables = [
                var for var in input_variables if var not in _partial_variables
            ]

        return cls(
            input_variables=input_variables,
            template=template,
            template_format=template_format,  # type: ignore[arg-type]
            partial_variables=_partial_variables,
            **kwargs,
        )
