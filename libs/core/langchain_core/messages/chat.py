from typing import Any, List, Literal

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)


class ChatMessage(BaseMessage):
    """Message that can be assigned an arbitrary speaker (i.e. role)."""

    role: str
    """The speaker / role of the Message."""

    type: Literal["chat"] = "chat"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


ChatMessage.update_forward_refs()


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """Chat Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ChatMessageChunk"] = "ChatMessageChunk"  # type: ignore

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                raise ValueError(
                    "Cannot concatenate ChatMessageChunks with different roles."
                )

            return self.__class__(
                role=self.role,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        elif isinstance(other, BaseMessageChunk):
            return self.__class__(
                role=self.role,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        else:
            return super().__add__(other)
