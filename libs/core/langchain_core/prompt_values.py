"""**Prompt values** for language model prompts.

Prompt values are used to represent different pieces of prompts. They can be used to
represent text, images, or chat message pieces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, cast

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
        return [HumanMessage(content=[cast("dict", self.image_url)])]


class ChatPromptValueConcrete(ChatPromptValue):
    """Chat prompt value which explicitly lists out the message types it accepts.

    For use in external schemas.
    """

    messages: Sequence[AnyMessage]
    """Sequence of messages."""

    type: Literal["ChatPromptValueConcrete"] = "ChatPromptValueConcrete"
