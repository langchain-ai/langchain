"""**Prompt values** for language model prompts.

Prompt values are used to represent different pieces of prompts.
They can be used to represent text, images, or chat message pieces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, Union, cast

from typing_extensions import TypedDict, overload

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.messages import content_blocks as types
from langchain_core.v1.messages import AIMessage as AIMessageV1
from langchain_core.v1.messages import HumanMessage as HumanMessageV1
from langchain_core.v1.messages import MessageV1, ResponseMetadata
from langchain_core.v1.messages import SystemMessage as SystemMessageV1
from langchain_core.v1.messages import ToolMessage as ToolMessageV1


def _convert_to_v1(message: BaseMessage) -> MessageV1:
    """Best-effort conversion of a V0 AIMessage to V1."""
    if isinstance(message.content, str):
        content: list[types.ContentBlock] = []
        if message.content:
            content = [{"type": "text", "text": message.content}]
    else:
        content = []
        for block in message.content:
            if isinstance(block, str):
                content.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                content.append(cast("types.ContentBlock", block))
            else:
                pass

    if isinstance(message, HumanMessage):
        return HumanMessageV1(content=content)
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            content.append(tool_call)
        return AIMessageV1(
            content=content,
            usage_metadata=message.usage_metadata,
            response_metadata=cast("ResponseMetadata", message.response_metadata),
            tool_calls=message.tool_calls,
        )
    if isinstance(message, SystemMessage):
        return SystemMessageV1(content=content)
    if isinstance(message, ToolMessage):
        return ToolMessageV1(
            tool_call_id=message.tool_call_id,
            content=content,
            artifact=message.artifact,
        )
    error_message = f"Unsupported message type: {type(message)}"
    raise TypeError(error_message)


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
    ChatModel inputs.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable. Defaults to True."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        This is used to determine the namespace of the object when serializing.
        Defaults to ["langchain", "schema", "prompt"].
        """
        return ["langchain", "schema", "prompt"]

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @overload
    def to_messages(
        self, message_version: Literal["v0"] = "v0"
    ) -> list[BaseMessage]: ...

    @overload
    def to_messages(self, message_version: Literal["v1"]) -> list[MessageV1]: ...

    @abstractmethod
    def to_messages(
        self, message_version: Literal["v0", "v1"] = "v0"
    ) -> Union[Sequence[BaseMessage], Sequence[MessageV1]]:
        """Return prompt as a list of Messages."""


class StringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    """Prompt text."""
    type: Literal["StringPromptValue"] = "StringPromptValue"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        This is used to determine the namespace of the object when serializing.
        Defaults to ["langchain", "prompts", "base"].
        """
        return ["langchain", "prompts", "base"]

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    @overload
    def to_messages(
        self, message_version: Literal["v0"] = "v0"
    ) -> list[BaseMessage]: ...

    @overload
    def to_messages(self, message_version: Literal["v1"]) -> list[MessageV1]: ...

    def to_messages(
        self, message_version: Literal["v0", "v1"] = "v0"
    ) -> Union[Sequence[BaseMessage], Sequence[MessageV1]]:
        """Return prompt as messages."""
        if message_version == "v1":
            return [HumanMessageV1(content=self.text)]
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

    @overload
    def to_messages(
        self, message_version: Literal["v0"] = "v0"
    ) -> list[BaseMessage]: ...

    @overload
    def to_messages(self, message_version: Literal["v1"]) -> list[MessageV1]: ...

    def to_messages(
        self, message_version: Literal["v0", "v1"] = "v0"
    ) -> Union[Sequence[BaseMessage], Sequence[MessageV1]]:
        """Return prompt as a list of messages.

        Args:
            message_version: The output version, either "v0" (default) or "v1".
        """
        if message_version == "v1":
            return [_convert_to_v1(m) for m in self.messages]
        return list(self.messages)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        This is used to determine the namespace of the object when serializing.
        Defaults to ["langchain", "prompts", "chat"].
        """
        return ["langchain", "prompts", "chat"]


class ImageURL(TypedDict, total=False):
    """Image URL."""

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image. Defaults to "auto".
    Can be "auto", "low", or "high"."""

    url: str
    """Either a URL of the image or the base64 encoded image data."""


class ImagePromptValue(PromptValue):
    """Image prompt value."""

    image_url: ImageURL
    """Image URL."""
    type: Literal["ImagePromptValue"] = "ImagePromptValue"

    def to_string(self) -> str:
        """Return prompt (image URL) as string."""
        return self.image_url["url"]

    @overload
    def to_messages(
        self, message_version: Literal["v0"] = "v0"
    ) -> list[BaseMessage]: ...

    @overload
    def to_messages(self, message_version: Literal["v1"]) -> list[MessageV1]: ...

    def to_messages(
        self, message_version: Literal["v0", "v1"] = "v0"
    ) -> Union[Sequence[BaseMessage], Sequence[MessageV1]]:
        """Return prompt (image URL) as messages."""
        if message_version == "v1":
            block: types.ImageContentBlock = {
                "type": "image",
                "url": self.image_url["url"],
            }
            if "detail" in self.image_url:
                block["detail"] = self.image_url["detail"]
            return [HumanMessageV1(content=[block])]
        return [HumanMessage(content=[cast("dict", self.image_url)])]


class ChatPromptValueConcrete(ChatPromptValue):
    """Chat prompt value which explicitly lists out the message types it accepts.

    For use in external schemas.
    """

    messages: Sequence[AnyMessage]
    """Sequence of messages."""

    type: Literal["ChatPromptValueConcrete"] = "ChatPromptValueConcrete"
