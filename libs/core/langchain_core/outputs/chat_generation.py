from __future__ import annotations

from typing import Literal, Union

from pydantic import model_validator
from typing_extensions import Self

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

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
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
        try:
            text = ""
            if isinstance(self.message.content, str):
                text = self.message.content
            # HACK: Assumes text in content blocks in OpenAI format.
            # Uses first text block.
            elif isinstance(self.message.content, list):
                for block in self.message.content:
                    if isinstance(block, str):
                        text = block
                        break
                    elif isinstance(block, dict) and "text" in block:
                        text = block["text"]
                        break
                    else:
                        pass
            else:
                pass
            self.text = text
        except (KeyError, AttributeError) as e:
            msg = "Error while initializing ChatGeneration"
            raise ValueError(msg) from e
        return self

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output"]


class ChatGenerationChunk(ChatGeneration):
    """ChatGeneration chunk, which can be concatenated with other
    ChatGeneration chunks.
    """

    message: BaseMessageChunk
    """The message chunk output by the chat model."""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output"]

    def __add__(
        self, other: Union[ChatGenerationChunk, list[ChatGenerationChunk]]
    ) -> ChatGenerationChunk:
        if isinstance(other, ChatGenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info or None,
            )
        elif isinstance(other, list) and all(
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
        else:
            msg = (
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )
            raise TypeError(msg)
