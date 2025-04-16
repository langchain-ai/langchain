"""Message prompt templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.load import Serializable
from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    get_template_variables,
)
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable.

        Returns: True.
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Default namespace is ["langchain", "prompts", "chat"].
        """
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variables.
        """

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """Print a human-readable representation."""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Combine two prompt templates.

        Args:
            other: Another prompt template.

        Returns:
            Combined prompt template.
        """
        from langchain_core.prompts.chat import ChatPromptTemplate

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other


class _DictMessagePromptTemplate(BaseMessagePromptTemplate):
    """Template represented by a dict that recursively fills input vars in string vals.

    Special handling of image_url dicts to load local paths. These look like:
    ``{"type": "image_url", "image_url": {"path": "..."}}``
    """

    template: dict[str, Any]
    template_format: Literal["f-string", "mustache"]

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        msg_dict = _insert_input_variables(self.template, kwargs, self.template_format)
        return convert_to_messages([msg_dict])

    @property
    def input_variables(self) -> list[str]:
        return _get_input_variables(self.template, self.template_format)

    @property
    def _prompt_type(self) -> str:
        return "message-dict-prompt"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_core", "prompts", "message"]

    def format(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format the prompt with the inputs."""
        return _insert_input_variables(self.template, kwargs, self.template_format)


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
            # No longer support loading local images.
            if k == "image_url" and "path" in v:
                msg = (
                    "Specifying image inputs via file path in environments with "
                    "user-input paths is a security vulnerability. Out of an abundance "
                    "of caution, the utility has been removed to prevent possible "
                    "misuse."
                )
                raise ValueError(msg)
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
    return formatted
