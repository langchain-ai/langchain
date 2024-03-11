from typing import Any, List, Literal, Optional

from langchain_core.load import Serializable
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content


class ToolCall(Serializable):
    name: str
    args: dict
    id: Optional[str] = None


class ToolCallsMessage(AIMessage):
    tool_calls: List[ToolCall]


ToolCallsMessage.update_forward_refs()


class ToolOutputMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model."""

    tool_call_id: str
    """Tool call that this message is responding to."""

    type: Literal["tool"] = "tool"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


ToolOutputMessage.update_forward_refs()
ToolMessage = ToolOutputMessage


class ToolOutputMessageChunk(ToolOutputMessage, BaseMessageChunk):
    """Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolOutputMessageChunk"] = "ToolOutputMessageChunk"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolOutputMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                raise ValueError(
                    "Cannot concatenate ToolOutputMessageChunks with different names."
                )

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


ToolMessageChunk = ToolOutputMessageChunk
