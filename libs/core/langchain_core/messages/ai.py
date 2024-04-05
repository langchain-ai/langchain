from json import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional
import warnings

from langchain_core.load import Serializable
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import parse_partial_json


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


class AIMessage(BaseMessage):
    """Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    tool_calls: Optional[List[ToolCall]] = None
    """If provided, tool calls associated with the message."""

    type: Literal["ai"] = "ai"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    @root_validator
    def _backwards_compat_tool_calls(cls, values: dict) -> dict:
        raw_tool_calls = values.get("additional_kwargs", {}).get("tool_calls")
        tool_calls = values.get("tool_calls") or values.get("tool_call_chunks")
        if raw_tool_calls and not tool_calls:
            warnings.warn(
                "You appear to be using an old tool calling model, please upgrade "
                "your packages to versions that set message tool calls."
            )
        # TODO: best-effort parsing
        return values

AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    tool_call_chunks: Optional[List[ToolCallChunk]] = None
    """If provided, tool call chunks associated with the message."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    @root_validator()
    def init_tool_calls(cls, values: dict) -> dict:
        if values["tool_calls"] is not None:
            raise ValueError(
                "tool_calls cannot be set on AIMessageChunk, it is derived "
                "from tool_call_chunks."
            )
        if not values["tool_call_chunks"]:
            values["tool_calls"] = values["tool_call_chunks"]
            return values
        tool_calls = []
        for chunk in values["tool_call_chunks"]:
            try:
                args_ = parse_partial_json(chunk.args)
                args_ = args_ if isinstance(args_, dict) else {}
            except (JSONDecodeError, TypeError):  # None args raise TypeError
                args_ = {}
            if not isinstance(args_, dict):
                raise ValueError(f"{args_=}")
            tool_calls.append(
                ToolCall(
                    name=chunk.name or "", args=args_, index=chunk.index, id=chunk.id
                )
            )
        values["tool_calls"] = tool_calls
        return values

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            content = merge_content(self.content, other.content)
            additional_kwargs = merge_dicts(
                self.additional_kwargs, other.additional_kwargs
            )
            response_metadata = merge_dicts(
                self.response_metadata, other.response_metadata
            )

            # Merge tool call chunks
            if self.tool_call_chunks or other.tool_call_chunks:
                raw_tool_calls = merge_lists(
                    [tc.dict() for tc in self.tool_call_chunks or []],
                    [tc.dict() for tc in other.tool_call_chunks or []],
                )
                if raw_tool_calls:
                    tool_call_chunks = [
                        ToolCallChunk(**rtc) for rtc in raw_tool_calls
                    ]
                else:
                    tool_call_chunks = None
            else:
                tool_call_chunks = None

            return self.__class__(
                example=self.example,
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_call_chunks,
                response_metadata=response_metadata,
                id=self.id,
            )

        return super().__add__(other)
