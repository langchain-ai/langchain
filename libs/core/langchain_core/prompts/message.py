"""Message prompt templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence

from langchain_core.load import Serializable
from langchain_core.messages import BaseMessage, convert_to_messages
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    get_template_variables,
)
from langchain_core.utils.image import image_to_data_url
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessagePromptTemplate(Serializable, ABC):
    """Base class for message prompt templates."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return True if the class is serializable, else False.

        Returns:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format messages from kwargs. Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """

    async def aformat_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Async format messages from kwargs.
        Should return a list of BaseMessages.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variables.
        """

    def pretty_repr(self, html: bool = False) -> str:
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

        prompt = ChatPromptTemplate(messages=[self])  # type: ignore[call-arg]
        return prompt + other


class _MessageDictPromptTemplate(BaseMessagePromptTemplate):
    """Template represented by a dict that looks for input vars in all leaf vals.

    Special handling of any dict value that contains "type": "image_url".
    """

    template: Dict[str, Any]
    template_format: Literal["f-string", "mustache"]

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        msg_dict = _insert_input_variables(self.template, kwargs, self.template_format)
        return convert_to_messages([msg_dict])

    @property
    def input_variables(self) -> List[str]:
        return _get_input_variables(self.template, self.template_format)

    @property
    def _prompt_type(self) -> str:
        return "message-dict-prompt"


def _get_input_variables(
    template: dict, template_format: Literal["f-string", "mustache"]
) -> List[str]:
    input_variables = []
    for k, v in template.items():
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
    template: Dict[str, Any],
    inputs: Dict[str, Any],
    template_format: Literal["f-string", "mustache"],
) -> Dict[str, Any]:
    formatted = {}
    formatter = DEFAULT_FORMATTER_MAPPING[template_format]
    for k, v in template.items():
        if isinstance(v, str):
            formatted[k] = formatter(v, **inputs)
        elif isinstance(v, dict):
            # Special handling for loading local images.
            if k == "image_url" and "path" in v:
                formatted_path = formatter(v.pop("path"), **inputs)
                v["url"] = image_to_data_url(formatted_path)
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
