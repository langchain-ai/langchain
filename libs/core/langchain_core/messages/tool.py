from typing import Any, List, Literal, Optional

from langchain_core.load import Serializable
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts
from langchain_core.utils.tools import parse_tool_calls


class ToolCall(Serializable):
    name: str
    args: dict
    id: Optional[str] = None


class ToolCallsMessage(AIMessage):
    tool_calls: Optional[List[ToolCall]] = None


class ToolCallsMessageChunk(ToolCallsMessage, BaseMessageChunk):
    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolCallsMessageChunk"] = "ToolCallsMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolCallsMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate ToolCallsMessageChunks "
                    "with different example values."
                )

            additional_kwargs = merge_dicts(
                self.additional_kwargs, other.additional_kwargs
            )
            raw_tool_calls = additional_kwargs.get("tool_calls", [])
            tool_calls = [
                ToolCall(**tool_call)
                for tool_call in parse_tool_calls(
                    raw_tool_calls, partial=True, return_id=True
                )
            ]

            return self.__class__(
                example=self.example,
                content=merge_content(self.content, other.content),
                additional_kwargs=additional_kwargs,
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                tool_calls=tool_calls,
            )

        return super().__add__(other)


ToolCallsMessage.update_forward_refs()


class ToolMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model."""

    tool_call_id: str
    """Tool call that this message is responding to."""

    type: Literal["tool"] = "tool"


ToolMessage.update_forward_refs()


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                raise ValueError(
                    "Cannot concatenate ToolMessageChunks with different names."
                )

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )

        return super().__add__(other)
