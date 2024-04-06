import json
from typing import Any, Dict, List, Literal, Optional

from langchain_core.load import Serializable
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class ToolMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model."""

    tool_call_id: str
    """Tool call that this message is responding to."""
    # TODO: Add is_error param?
    # is_error: bool = False
    # """Whether the tool errored."""

    type: Literal["tool"] = "tool"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


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


class ToolCall(Serializable):
    """A call to a tool.

    Attributes:
        name: (str) the name of the tool to be called
        args: (dict) the arguments to the tool call
        id: (str) if provided, an identifier associated with the tool call
        index: (int) if provided, the index of the tool call in a sequence
            of content
    """

    name: str
    args: Dict[str, Any]
    id: Optional[str] = None
    index: Optional[int] = None


class ToolCallChunk(Serializable):
    """A chunk of a tool call (e.g., as part of a stream).

    When merging ToolCallChunks (e.g., via AIMessageChunk.__add__),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not None.

    Example:

    .. code-block:: python
        left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
        right_chunks = [ToolCallChunk(name=None, args='1}', index=0)]
        (
            AIMessageChunk(content="", tool_call_chunks=left_chunks)
            + AIMessageChunk(content="", tool_call_chunks=right_chunks)
        ).tool_call_chunks == [ToolCallChunk(name='foo', args='{"a":1}', index=0)]

    Attributes:
        name: (str) if provided, a substring of the name of the tool to be called
        args: (str) if provided, a JSON substring of the arguments to the tool call
        id: (str) if provided, a substring of an identifier for the tool call
        index: (int) if provided, the index of the tool call in a sequence
    """

    name: Optional[str] = None
    args: Optional[str] = None
    id: Optional[str] = None
    index: Optional[int] = None


class InvalidToolCall(Serializable):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    """

    name: Optional[str] = None
    args: Optional[str] = None
    id: Optional[str] = None
    error: Optional[str] = None


def default_tool_parser(raw_tool_calls: List[dict]) -> List[ToolCall]:
    """Best-effort parsing of tools."""
    tool_calls = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = json.loads(tool_call["function"]["arguments"])
            function_name = tool_call["function"]["name"]
        parsed = ToolCall(
            name=function_name or "",
            args=function_args or {},
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_calls.append(parsed)
    return tool_calls


def default_tool_chunk_parser(raw_tool_calls: List[dict]) -> List[ToolCallChunk]:
    """Best-effort parsing of tool chunks."""
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        parsed = ToolCallChunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks
