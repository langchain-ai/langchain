"""BasePrompt schema definition."""
from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Callable, Dict, List, Set

from langchain.formatting import formatter
from langchain.schema import BasePromptTemplate
from langchain.schema.messages import BaseMessage, HumanMessage
from langchain.schema.prompt import PromptValue


def jinja2_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using jinja2."""
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )

    return Template(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: List[str]) -> None:
    """
    Validate that the input variables are valid for the template.
    Issues an warning if missing or extra variables are found.

    Args:
        template: The template string.
        input_variables: The input variables.
    """
    input_variables_set = set(input_variables)
    valid_variables = _get_jinja2_variables_from_template(template)
    missing_variables = valid_variables - input_variables_set
    extra_variables = input_variables_set - valid_variables

    warning_message = ""
    if missing_variables:
        warning_message += f"Missing variables: {missing_variables} "

    if extra_variables:
        warning_message += f"Extra variables: {extra_variables}"

    if warning_message:
        warnings.warn(warning_message.strip())


def _get_jinja2_variables_from_template(template: str) -> Set[str]:
    try:
        from jinja2 import Environment, meta
    except ImportError:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )
    env = Environment()
    ast = env.parse(template)
    variables = meta.find_undeclared_variables(ast)
    return variables


DEFAULT_FORMATTER_MAPPING: Dict[str, Callable] = {
    "f-string": formatter.format,
    "jinja2": jinja2_formatter,
}

DEFAULT_VALIDATOR_MAPPING: Dict[str, Callable] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
}


def check_valid_template(
    template: str, template_format: str, input_variables: List[str]
) -> None:
    """Check that template string is valid."""
    if template_format not in DEFAULT_FORMATTER_MAPPING:
        valid_formats = list(DEFAULT_FORMATTER_MAPPING)
        raise ValueError(
            f"Invalid template format. Got `{template_format}`;"
            f" should be one of {valid_formats}"
        )
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
        validator_func(template, input_variables)
    except KeyError as e:
        raise ValueError(
            "Invalid prompt schema; check for mismatched or missing input parameters. "
            + str(e)
        )


class StringPromptValue(PromptValue):
    text: str

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=self.text)]


class StringPromptTemplate(BasePromptTemplate, ABC):
    """String prompt should expose the format method, returning a prompt."""

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""
        return StringPromptValue(text=self.format(**kwargs))
