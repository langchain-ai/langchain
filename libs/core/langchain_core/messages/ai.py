from typing import Any, Literal

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)


class AIMessage(BaseMessage):
    """A Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["ai"] = "ai"


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """A Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            return self.__class__(
                example=self.example,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)
