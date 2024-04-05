from json import JSONDecodeError
from typing import Any, List, Literal, Optional

from langchain_core.load import Serializable
from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import parse_partial_json


class AIMessage(BaseMessage):
    """Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["ai"] = "ai"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

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

            if isinstance(other, AIToolCallsMessageChunk):
                return AIToolCallsMessageChunk(
                    example=self.example,
                    content=content,
                    additional_kwargs=additional_kwargs,
                    response_metadata=response_metadata,
                    tool_call_chunks=other.tool_call_chunks,
                    id=self.id,
                )

            return self.__class__(
                example=self.example,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
                id=self.id,
            )

        return super().__add__(other)


class ToolCall(Serializable):
    name: str
    args: dict
    id: Optional[str] = None
    index: Optional[int] = None


class ToolCallChunk(Serializable):
    name: Optional[str] = None
    args: Optional[str] = None
    id: Optional[str] = None
    index: Optional[int] = None


class AIToolCallsMessage(AIMessage):
    tool_calls: Optional[List[ToolCall]] = None
    type: Literal["tool_calls"] = "tool_calls"  # type: ignore[assignment] # noqa: E501


class AIToolCallsMessageChunk(AIToolCallsMessage, AIMessageChunk):
    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIToolCallsMessageChunk"] = "AIToolCallsMessageChunk"  # type: ignore[assignment] # noqa: E501
    tool_call_chunks: Optional[List[ToolCallChunk]] = None

    @root_validator()
    def init_tool_calls(cls, values: dict) -> dict:
        if values["tool_calls"] is not None:
            raise ValueError(
                "tool_calls cannot be set on AIToolCallsMessageChunk, it is derived "
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
            except JSONDecodeError:
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
                    "Cannot concatenate AIToolCallsMessageChunks "
                    "with different example values."
                )

            tool_call_chunks: Optional[List] = None
            if isinstance(other, AIToolCallsMessageChunk):
                if self.tool_call_chunks or other.tool_call_chunks:
                    raw_tool_calls = merge_lists(
                        [tc.dict() for tc in self.tool_call_chunks or []],
                        [tc.dict() for tc in other.tool_call_chunks or []],
                    )
                    tool_call_chunks = [ToolCallChunk(**rtc) for rtc in raw_tool_calls]

            return self.__class__(
                example=self.example,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                tool_call_chunks=tool_call_chunks,
                id=self.id,
            )

        return super().__add__(other)


AIToolCallsMessage.update_forward_refs()
