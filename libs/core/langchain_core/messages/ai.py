from typing import Any, Dict, List, Literal, Union

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.messages.tool import (
    InvalidToolCall,
    ToolCall,
    ToolCallChunk,
    default_tool_chunk_parser,
    default_tool_parser,
)
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import (
    parse_partial_json,
)


class AIMessage(BaseMessage):
    """Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    tool_calls: List[ToolCall] = []
    """If provided, tool calls associated with the message."""
    invalid_tool_calls: List[InvalidToolCall] = []
    """If provided, tool calls with parsing errors associated with the message."""

    type: Literal["ai"] = "ai"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @root_validator()
    def _backwards_compat_tool_calls(cls, values: dict) -> dict:
        raw_tool_calls = values.get("additional_kwargs", {}).get("tool_calls")
        tool_calls = (
            values.get("tool_calls")
            or values.get("invalid_tool_calls")
            or values.get("tool_call_chunks")
        )
        if raw_tool_calls and not tool_calls:
            try:
                if issubclass(cls, AIMessageChunk):  # type: ignore
                    values["tool_call_chunks"] = default_tool_chunk_parser(
                        raw_tool_calls
                    )
                else:
                    tool_calls, invalid_tool_calls = default_tool_parser(raw_tool_calls)
                    values["tool_calls"] = tool_calls
                    values["invalid_tool_calls"] = invalid_tool_calls
            except Exception:
                pass
        return values

    def pretty_repr(self, html: bool = False) -> str:
        """Return a pretty representation of the message."""
        base = super().pretty_repr(html=html)
        lines = []

        def _format_tool_args(tc: Union[ToolCall, InvalidToolCall]) -> List[str]:
            lines = [
                f"  {tc.get('name', 'Tool')} ({tc.get('id')})",
                f" Call ID: {tc.get('id')}",
            ]
            if tc.get("error"):
                lines.append(f"  Error: {tc.get('error')}")
            lines.append("  Args:")
            args = tc.get("args")
            if isinstance(args, str):
                lines.append(f"    {args}")
            elif isinstance(args, dict):
                for arg, value in args.items():
                    lines.append(f"    {arg}: {value}")
            return lines

        if self.tool_calls:
            lines.append("Tool Calls:")
            for tc in self.tool_calls:
                lines.extend(_format_tool_args(tc))
        if self.invalid_tool_calls:
            lines.append("Invalid Tool Calls:")
            for itc in self.invalid_tool_calls:
                lines.extend(_format_tool_args(itc))
        return (base.strip() + "\n" + "\n".join(lines)).strip()


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    tool_call_chunks: List[ToolCallChunk] = []
    """If provided, tool call chunks associated with the message."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @root_validator()
    def init_tool_calls(cls, values: dict) -> dict:
        if not values["tool_call_chunks"]:
            values["tool_calls"] = []
            values["invalid_tool_calls"] = []
            return values
        tool_calls = []
        invalid_tool_calls = []
        for chunk in values["tool_call_chunks"]:
            try:
                args_ = parse_partial_json(chunk["args"])
                if isinstance(args_, dict):
                    tool_calls.append(
                        ToolCall(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )
                else:
                    raise ValueError("Malformed args.")
            except Exception:
                invalid_tool_calls.append(
                    InvalidToolCall(
                        name=chunk["name"],
                        args=chunk["args"],
                        id=chunk["id"],
                        error=None,
                    )
                )
        values["tool_calls"] = tool_calls
        values["invalid_tool_calls"] = invalid_tool_calls
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
                    self.tool_call_chunks,
                    other.tool_call_chunks,
                )
                if raw_tool_calls:
                    tool_call_chunks = [
                        ToolCallChunk(
                            name=rtc.get("name"),
                            args=rtc.get("args"),
                            index=rtc.get("index"),
                            id=rtc.get("id"),
                        )
                        for rtc in raw_tool_calls
                    ]
                else:
                    tool_call_chunks = []
            else:
                tool_call_chunks = []

            return self.__class__(
                example=self.example,
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_call_chunks,
                response_metadata=response_metadata,
                id=self.id,
            )

        return super().__add__(other)
