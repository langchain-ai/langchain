"""**Prompt values** for language model prompts.

Prompt values are used to represent different pieces of prompts. They can be used to
represent text, images, or chat message pieces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, cast

from typing_extensions import TypedDict

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    `PromptValues` can be converted to both LLM (pure text-generation) inputs and
    chat model inputs.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "prompt"]`
        """
        return ["langchain", "schema", "prompt"]

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> list[BaseMessage]:
        """Return prompt as a list of messages."""

    def __getstate__(self) -> dict[Any, Any]:
        """Leave the tracing-only prompt template info out of pickles."""
        state = super().__getstate__()
        if _PROMPT_TEMPLATE_INFO_ATTR in state.get("__dict__", {}):
            state["__dict__"] = {
                key: value
                for key, value in state["__dict__"].items()
                if key != _PROMPT_TEMPLATE_INFO_ATTR
            }
        return state


_PROMPT_TEMPLATE_INFO_ATTR = "_lc_prompt_template_info"


def _set_prompt_template_info(prompt_value: PromptValue, info: dict[str, Any]) -> None:
    """Record which prompt template produced `prompt_value`, for tracing.

    The info is stored outside the pydantic fields, so equality, serialization
    (`dumpd`, `model_dump`) and the JSON schema of the prompt value are unchanged.
    Language models read it back with `_get_prompt_template_info` and pass it to
    their callbacks.
    """
    object.__setattr__(prompt_value, _PROMPT_TEMPLATE_INFO_ATTR, info)


def _get_prompt_template_info(value: object) -> dict[str, Any] | None:
    """Return the info recorded by `_set_prompt_template_info`, if any."""
    if isinstance(value, PromptValue):
        return value.__dict__.get(_PROMPT_TEMPLATE_INFO_ATTR)
    return None


def _pop_prompt_template_option(
    model_input: object, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return the callback `options` entry describing the template behind the input.

    The info comes from the prompt value the template produced, or from an explicit
    `ls_prompt_template` keyword argument, which is how `generate_prompt` hands it to
    `generate`. That argument is always removed from `kwargs`, so it never reaches the
    provider, the invocation params or the cache key.

    Args:
        model_input: The input passed to the model, before conversion to messages.
        kwargs: The model call's keyword arguments; modified in place.

    Returns:
        `{"ls_prompt_template": info}`, or `{}` when the input wasn't produced by a
            prompt template.
    """
    info = kwargs.pop("ls_prompt_template", None) or _get_prompt_template_info(
        model_input
    )
    return {"ls_prompt_template": info} if info else {}


class StringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    """Prompt text."""

    type: Literal["StringPromptValue"] = "StringPromptValue"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "base"]`
        """
        return ["langchain", "prompts", "base"]

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> list[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=self.text)]


class ChatPromptValue(PromptValue):
    """Chat prompt value.

    A type of a prompt value that is built from messages.
    """

    messages: Sequence[BaseMessage]
    """List of messages."""

    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> list[BaseMessage]:
        """Return prompt as a list of messages."""
        return list(self.messages)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]


class ImageURL(TypedDict, total=False):
    """Image URL for multimodal model inputs (OpenAI format).

    Represents the inner `image_url` object in OpenAI's Chat Completion API format. This
    is used by `ImagePromptTemplate` and `ChatPromptTemplate`.

    See Also:
        `ImageContentBlock`: LangChain's provider-agnostic image format used in message
        content blocks. Use `ImageContentBlock` when working with the standardized
        message format across different providers.

    Note:
        The `detail` field values are not validated locally. Invalid values
        will be rejected by the downstream API, allowing new valid values to
        be used without requiring a LangChain update.
    """

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image.

    Defaults to ``'auto'`` if not specified. Higher detail levels consume
    more tokens but provide better image understanding.
    """

    url: str
    """URL of the image or base64-encoded image data."""


class ImagePromptValue(PromptValue):
    """Image prompt value."""

    image_url: ImageURL
    """Image URL."""

    type: Literal["ImagePromptValue"] = "ImagePromptValue"

    def to_string(self) -> str:
        """Return prompt (image URL) as string."""
        return self.image_url.get("url", "")

    def to_messages(self) -> list[BaseMessage]:
        """Return prompt (image URL) as messages."""
        return [HumanMessage(content=[cast("dict[str, Any]", self.image_url)])]


class ChatPromptValueConcrete(ChatPromptValue):
    """Chat prompt value which explicitly lists out the message types it accepts.

    For use in external schemas.
    """

    messages: Sequence[AnyMessage]
    """Sequence of messages."""

    type: Literal["ChatPromptValueConcrete"] = "ChatPromptValueConcrete"
