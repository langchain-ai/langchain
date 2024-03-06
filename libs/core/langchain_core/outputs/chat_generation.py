from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts


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

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output"]


class ChatGenerationChunk(ChatGeneration):
    """ChatGeneration chunk, which can be concatenated with other
      ChatGeneration chunks.

    Attributes:
        message: The message chunk output by the chat model.
    """

    message: BaseMessageChunk
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment] # noqa: E501
    """Type is used exclusively for serialization purposes."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output"]

    def __add__(self, other: ChatGenerationChunk) -> ChatGenerationChunk:
        if isinstance(other, ChatGenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info or None,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )
