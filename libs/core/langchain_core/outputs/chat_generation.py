from __future__ import annotations

from typing import Any, Dict, Literal

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import root_validator


class ChatGeneration(Generation):
    """A single chat generation output."""

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
    message: BaseMessage
    """The message output by the chat model."""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes."""

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        try:
            values["text"] = values["message"].content
        except (KeyError, AttributeError) as e:
            raise ValueError("Error while initializing ChatGeneration") from e
        return values


class ChatGenerationChunk(ChatGeneration):
    """A ChatGeneration chunk, which can be concatenated with other
      ChatGeneration chunks.

    Attributes:
        message: The message chunk output by the chat model.
    """

    message: BaseMessageChunk
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment] # noqa: E501
    """Type is used exclusively for serialization purposes."""

    def __add__(self, other: ChatGenerationChunk) -> ChatGenerationChunk:
        if isinstance(other, ChatGenerationChunk):
            generation_info = (
                {**(self.generation_info or {}), **(other.generation_info or {})}
                if self.generation_info is not None or other.generation_info is not None
                else None
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )
