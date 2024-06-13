"""BasePrompt schema definition."""

from __future__ import annotations

import warnings
from abc import ABC
from string import Formatter
from typing import Any, Callable, Dict, List, Set, Tuple, Type

import langchain_core.utils.mustache as mustache
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.utils import get_colored_text
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env


def jinja2_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using jinja2.

    *Security warning*: As of LangChain 0.0.329, this method uses Jinja2's
        SandboxedEnvironment by default. However, this sand-boxing should
        be treated as a best-effort approach rather than a guarantee of security.
        Do not accept jinja2 templates from untrusted sources as they may lead
        to arbitrary Python code execution.

        https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    """
    try:
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
            "Please be cautious when using jinja2 templates. "
            "Do not expand jinja2 templates using unverified or user-controlled "
            "inputs as that can result in arbitrary Python code execution."
        )

    # This uses a sandboxed environment to prevent arbitrary code execution.
    # Jinja2 uses an opt-out rather than opt-in approach for sand-boxing.
    # Please treat this sand-boxing as a best-effort approach rather than
    # a guarantee of security.
    # We recommend to never use jinja2 templates with untrusted inputs.
    # https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    # approach not a guarantee of security.
    return SandboxedEnvironment().from_string(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: List[str]) -> None:
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


def mustache_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using mustache."""
    return mustache.render(template, kwargs)


def mustache_template_vars(
    template: str,
) -> Set[str]:
    """Get the variables from a mustache template."""
    vars: Set[str] = set()
    in_section = False
    for type, key in mustache.tokenize(template):
        if type == "end":
            in_section = False
        elif in_section:
            continue
        elif type in ("variable", "section") and key != ".":
            vars.add(key.split(".")[0])
            if type == "section":
                in_section = True
    return vars


Defs = Dict[str, "Defs"]


def mustache_schema(
    template: str,
) -> Type[BaseModel]:
    """Get the variables from a mustache template."""
    fields = set()
    prefix: Tuple[str, ...] = ()
    for type, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type == "end":
            prefix = prefix[: -key.count(".")]
        elif type == "section":
            prefix = prefix + tuple(key.split("."))
        elif type == "variable":
            fields.add(prefix + tuple(key.split(".")))
    defs: Defs = {}  # None means leaf node
    while fields:
        field = fields.pop()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current[field[-1]] = {}
    return _create_model_recursive("PromptInput", defs)


def _create_model_recursive(name: str, defs: Defs) -> Type:
    return create_model(  # type: ignore[call-overload]
        name,
        **{
            k: (_create_model_recursive(k, v), None) if v else (str, None)
            for k, v in defs.items()
        },
    )


DEFAULT_FORMATTER_MAPPING: Dict[str, Callable] = {
    "f-string": formatter.format,
    "mustache": mustache_formatter,
    "jinja2": jinja2_formatter,
}

DEFAULT_VALIDATOR_MAPPING: Dict[str, Callable] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
}


def check_valid_template(
    template: str, template_format: str, input_variables: List[str]
) -> None:
    """Check that template string is valid.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".
        input_variables: The input variables.

    Raises:
        ValueError: If the template format is not supported.
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


def get_template_variables(template: str, template_format: str) -> List[str]:
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
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "base"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        return StringPromptValue(text=await self.aformat(**kwargs))

    def pretty_repr(self, html: bool = False) -> str:
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
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201
