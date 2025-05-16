"""Chat generation output classes."""

from __future__ import annotations

from typing import Literal, Union

from pydantic import computed_field

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.utils._merge import merge_dicts


class ChatGeneration(Generation):
    """A single chat generation output.

    A subclass of Generation that represents the response from a chat model
    that generates chat messages.

    The `message` attribute is a structured representation of the chat message.
    Most of the time, the message will be of type `AIMessage`.

    Users working with chat models will usually access information via either
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks).
    """

    message: BaseMessage
    """The message output by the chat model."""

    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """Set the text attribute to be the contents of the message."""
        text_ = ""
        if isinstance(self.message.content, str):
            text_ = self.message.content
        # Assumes text in content blocks in OpenAI format.
        # Uses first text block.
        elif isinstance(self.message.content, list):
            for block in self.message.content:
                if isinstance(block, str):
                    text_ = block
                    break
                if isinstance(block, dict) and "text" in block:
                    text_ = block["text"]
                    break
        return text_


class ChatGenerationChunk(ChatGeneration):
    """ChatGeneration chunk.

    ChatGeneration chunks can be concatenated with other ChatGeneration chunks.
    """

    message: BaseMessageChunk
    """The message chunk output by the chat model."""

    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    def __add__(
        self, other: Union[ChatGenerationChunk, list[ChatGenerationChunk]]
    ) -> ChatGenerationChunk:
        """Concatenate two ChatGenerationChunks.

        Args:
            other: The other ChatGenerationChunk or list of ChatGenerationChunks to
                concatenate.
        """
        if isinstance(other, ChatGenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info or None,
            )
        if isinstance(other, list) and all(
            isinstance(x, ChatGenerationChunk) for x in other
        ):
            generation_info = merge_dicts(
                self.generation_info or {},
                *[chunk.generation_info for chunk in other if chunk.generation_info],
            )
            return ChatGenerationChunk(
                message=self.message + [chunk.message for chunk in other],
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)


def merge_chat_generation_chunks(
    chunks: list[ChatGenerationChunk],
) -> Union[ChatGenerationChunk, None]:
    """Merge a list of ChatGenerationChunks into a single ChatGenerationChunk."""
    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    return chunks[0] + chunks[1:]
