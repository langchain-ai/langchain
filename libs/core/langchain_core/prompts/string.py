"""BasePrompt schema definition."""

from __future__ import annotations

import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable

from pydantic import BaseModel, create_model

import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env


def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    """Format a template using jinja2.

    *Security warning*:
        As of LangChain 0.0.329, this method uses Jinja2's
        SandboxedEnvironment by default. However, this sand-boxing should
        be treated as a best-effort approach rather than a guarantee of security.
        Do not accept jinja2 templates from untrusted sources as they may lead
        to arbitrary Python code execution.

        https://jinja.palletsprojects.com/en/3.1.x/sandbox/

    Args:
        template: The template string.
        **kwargs: The variables to format the template with.

    Returns:
        The formatted string.

    Raises:
        ImportError: If jinja2 is not installed.
    """
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError as e:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
            "Please be cautious when using jinja2 templates. "
            "Do not expand jinja2 templates using unverified or user-controlled "
            "inputs as that can result in arbitrary Python code execution."
        ) from e

    # This uses a sandboxed environment to prevent arbitrary code execution.
    # Jinja2 uses an opt-out rather than opt-in approach for sand-boxing.
    # Please treat this sand-boxing as a best-effort approach rather than
    # a guarantee of security.
    # We recommend to never use jinja2 templates with untrusted inputs.
    # https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    # approach not a guarantee of security.
    return SandboxedEnvironment().from_string(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: list[str]) -> None:
    """
    Validate that the input variables are valid for the template.
    Issues a warning if missing or extra variables are found.

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
        warnings.warn(warning_message.strip(), stacklevel=7)


def _get_jinja2_variables_from_template(template: str) -> set[str]:
    try:
        from jinja2 import Environment, meta
    except ImportError as e:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        ) from e
    env = Environment()
    ast = env.parse(template)
    variables = meta.find_undeclared_variables(ast)
    return variables


def mustache_formatter(template: str, /, **kwargs: Any) -> str:
    """Format a template using mustache.

    Args:
        template: The template string.
        **kwargs: The variables to format the template with.

    Returns:
        The formatted string.
    """
    return mustache.render(template, kwargs)


def mustache_template_vars(
    template: str,
) -> set[str]:
    """Get the variables from a mustache template.

    Args:
        template: The template string.

    Returns:
        The variables from the template.
    """
    vars: set[str] = set()
    section_depth = 0
    for type, key in mustache.tokenize(template):
        if type == "end":
            section_depth -= 1
        elif (
            type in ("variable", "section", "inverted section", "no escape")
            and key != "."
            and section_depth == 0
        ):
            vars.add(key.split(".")[0])
        if type in ("section", "inverted section"):
            section_depth += 1
    return vars


Defs = dict[str, "Defs"]


def mustache_schema(
    template: str,
) -> type[BaseModel]:
    """Get the variables from a mustache template.

    Args:
        template: The template string.

    Returns:
        The variables from the template as a Pydantic model.
    """
    fields = {}
    prefix: tuple[str, ...] = ()
    section_stack: list[tuple[str, ...]] = []
    for type, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type == "end":
            if section_stack:
                prefix = section_stack.pop()
        elif type in ("section", "inverted section"):
            section_stack.append(prefix)
            prefix = prefix + tuple(key.split("."))
            fields[prefix] = False
        elif type in ("variable", "no escape"):
            fields[prefix + tuple(key.split("."))] = True
    defs: Defs = {}  # None means leaf node
    while fields:
        field, is_leaf = fields.popitem()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current.setdefault(field[-1], "" if is_leaf else {})  # type: ignore[arg-type]
    return _create_model_recursive("PromptInput", defs)


def _create_model_recursive(name: str, defs: Defs) -> type:
    return create_model(  # type: ignore[call-overload]
        name,
        **{
            k: (_create_model_recursive(k, v), None) if v else (type(v), None)
            for k, v in defs.items()
        },
    )


DEFAULT_FORMATTER_MAPPING: dict[str, Callable] = {
    "f-string": formatter.format,
    "mustache": mustache_formatter,
    "jinja2": jinja2_formatter,
}

DEFAULT_VALIDATOR_MAPPING: dict[str, Callable] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
}


def check_valid_template(
    template: str, template_format: str, input_variables: list[str]
) -> None:
    """Check that template string is valid.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".
        input_variables: The input variables.

    Raises:
        ValueError: If the template format is not supported.
        ValueError: If the prompt schema is invalid.
    """
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
    except KeyError as exc:
        raise ValueError(
            f"Invalid template format {template_format!r}, should be one of"
            f" {list(DEFAULT_FORMATTER_MAPPING)}."
        ) from exc
    try:
        validator_func(template, input_variables)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "Invalid prompt schema; check for mismatched or missing input parameters"
            f" from {input_variables}."
        ) from exc


def get_template_variables(template: str, template_format: str) -> list[str]:
    """Get the variables from the template.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".

    Returns:
        The variables from the template.

    Raises:
        ValueError: If the template format is not supported.
    """
    if template_format == "jinja2":
        # Get the variables for the template
        input_variables = _get_jinja2_variables_from_template(template)
    elif template_format == "f-string":
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
    elif template_format == "mustache":
        input_variables = mustache_template_vars(template)
    else:
        raise ValueError(f"Unsupported template format: {template_format}")

    return sorted(input_variables)


class StringPromptTemplate(BasePromptTemplate, ABC):
    """String prompt that exposes the format method, returning a prompt."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "base"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return StringPromptValue(text=await self.aformat(**kwargs))

    def pretty_repr(self, html: bool = False) -> str:
        """Get a pretty representation of the prompt.

        Args:
            html: Whether to return an HTML-formatted string.

        Returns:
            A pretty representation of the prompt.
        """
        # TODO: handle partials
        dummy_vars = {
            input_var: "{" + f"{input_var}" + "}" for input_var in self.input_variables
        }
        if html:
            dummy_vars = {
                k: get_colored_text(v, "yellow") for k, v in dummy_vars.items()
            }
        return self.format(**dummy_vars)

    def pretty_print(self) -> None:
        """Print a pretty representation of the prompt."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201
