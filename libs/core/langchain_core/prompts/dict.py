"""Dict prompt template."""

import warnings
from functools import cached_property
from typing import Any, Literal

from typing_extensions import override

from langchain_core.load import dumpd
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    get_template_variables,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config


class DictPromptTemplate(RunnableSerializable[dict, dict]):
    """Template represented by a dict.

    Recognizes variables in f-string or mustache formatted string dict values. Does NOT
    recognize variables in dict keys. Applies recursively.
    """

    template: dict[str, Any]
    template_format: Literal["f-string", "mustache"]

    @property
    def input_variables(self) -> list[str]:
        """Template input variables."""
        return _get_input_variables(self.template, self.template_format)

    def format(self, **kwargs: Any) -> dict[str, Any]:
        """Format the prompt with the inputs.

        Returns:
            A formatted dict.
        """
        return _insert_input_variables(self.template, kwargs, self.template_format)

    async def aformat(self, **kwargs: Any) -> dict[str, Any]:
        """Format the prompt with the inputs.

        Returns:
            A formatted dict.
        """
        return self.format(**kwargs)

    @override
    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict:
        return self._call_with_config(
            lambda x: self.format(**x),
            input,
            ensure_config(config),
            run_type="prompt",
            serialized=self._serialized,
            **kwargs,
        )

    @property
    def _prompt_type(self) -> str:
        return "dict-prompt"

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        return dumpd(self)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain_core", "prompts", "dict"]`
        """
        return ["langchain_core", "prompts", "dict"]

    def pretty_repr(self, *, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        raise NotImplementedError


def _get_input_variables(
    template: dict, template_format: Literal["f-string", "mustache"]
) -> list[str]:
    input_variables = []
    for v in template.values():
        if isinstance(v, str):
            input_variables += get_template_variables(v, template_format)
        elif isinstance(v, dict):
            input_variables += _get_input_variables(v, template_format)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    input_variables += get_template_variables(x, template_format)
                elif isinstance(x, dict):
                    input_variables += _get_input_variables(x, template_format)
    return list(set(input_variables))


def _insert_input_variables(
    template: dict[str, Any],
    inputs: dict[str, Any],
    template_format: Literal["f-string", "mustache"],
) -> dict[str, Any]:
    formatted = {}
    formatter = DEFAULT_FORMATTER_MAPPING[template_format]
    for k, v in template.items():
        if isinstance(v, str):
            formatted[k] = formatter(v, **inputs)
        elif isinstance(v, dict):
            if k == "image_url" and "path" in v:
                msg = (
                    "Specifying image inputs via file path in environments with "
                    "user-input paths is a security vulnerability. Out of an abundance "
                    "of caution, the utility has been removed to prevent possible "
                    "misuse."
                )
                warnings.warn(msg, stacklevel=2)
            formatted[k] = _insert_input_variables(v, inputs, template_format)
        elif isinstance(v, (list, tuple)):
            formatted_v = []
            for x in v:
                if isinstance(x, str):
                    formatted_v.append(formatter(x, **inputs))
                elif isinstance(x, dict):
                    formatted_v.append(
                        _insert_input_variables(x, inputs, template_format)
                    )
            formatted[k] = type(v)(formatted_v)
        else:
            formatted[k] = v
    return formatted
