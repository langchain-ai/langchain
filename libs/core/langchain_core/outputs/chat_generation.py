"""Chat generation output classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.utils._merge import merge_dicts

if TYPE_CHECKING:
    from typing_extensions import Self


class ChatGeneration(Generation):
    """A single chat generation output.

    A subclass of `Generation` that represents the response from a chat model that
    generates chat messages.

    The `message` attribute is a structured representation of the chat message. Most of
    the time, the message will be of type `AIMessage`.

    Users working with chat models will usually access information via either
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available via
    callbacks).
    """

    text: str = ""
    """The text contents of the output message.

    !!! warning "SHOULD NOT BE SET DIRECTLY!"

    """
    message: BaseMessage
    """The message output by the chat model."""

    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    @model_validator(mode="after")
    def set_text(self) -> Self:
        """Set the text attribute to be the contents of the message.

        Args:
            values: The values of the object.

        Returns:
            The values of the object with the text attribute set.

        Raises:
            ValueError: If the message is not a string or a list.
        """
        # Check for legacy blocks with "text" key but no "type" field.
        # Otherwise, delegate to `message.text`.
        if isinstance(self.message.content, list):
            has_legacy_blocks = any(
                isinstance(block, dict)
                and "text" in block
                and block.get("type") is None
                for block in self.message.content
            )

            if has_legacy_blocks:
                blocks = []
                for block in self.message.content:
                    if isinstance(block, str):
                        blocks.append(block)
                    elif isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text" or (
                            block_type is None and "text" in block
                        ):
                            blocks.append(block.get("text", ""))
                self.text = "".join(blocks)
            else:
                self.text = self.message.text
        else:
            self.text = self.message.text

        return self


class ChatGenerationChunk(ChatGeneration):
    """`ChatGeneration` chunk.

    `ChatGeneration` chunks can be concatenated with other `ChatGeneration` chunks.
    """

    message: BaseMessageChunk
    """The message chunk output by the chat model."""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional

    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    def __add__(
        self, other: ChatGenerationChunk | list[ChatGenerationChunk]
    ) -> ChatGenerationChunk:
        """Concatenate two `ChatGenerationChunk`s.

        Args:
            other: The other `ChatGenerationChunk` or list of `ChatGenerationChunk` to
                concatenate.

        Raises:
            TypeError: If other is not a `ChatGenerationChunk` or list of
                `ChatGenerationChunk`.

        Returns:
            A new `ChatGenerationChunk` concatenated from self and other.
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
) -> ChatGenerationChunk | None:
    """Merge a list of `ChatGenerationChunk`s into a single `ChatGenerationChunk`.

    Args:
        chunks: A list of `ChatGenerationChunk` to merge.

    Returns:
        A merged `ChatGenerationChunk`, or `None` if the input list is empty.
    """
    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    return chunks[0] + chunks[1:]
