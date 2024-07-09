import json
from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict

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


class UsageMetadata(TypedDict):
    """Usage metadata for a message, such as token counts.

    This is a standard representation of token usage that is consistent across models.

    Example:

        .. code-block:: python

            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30
            }
    """

    input_tokens: int
    """Count of input (or prompt) tokens."""
    output_tokens: int
    """Count of output (or completion) tokens."""
    total_tokens: int
    """Total token count."""


class AIMessage(BaseMessage):
    """Message from an AI.

    AIMessage is returned from a chat model as a response to a prompt.

    This message represents the output of the model and consists of both
    the raw output as returned by the model together standardized fields
    (e.g., tool calls, usage metadata) added by the LangChain framework.
    """

    example: bool = False
    """Use to denote that a message is part of an example conversation.
    
    At the moment, this is ignored by most models. Usage is discouraged.
    """

    tool_calls: List[ToolCall] = []
    """If provided, tool calls associated with the message."""
    invalid_tool_calls: List[InvalidToolCall] = []
    """If provided, tool calls with parsing errors associated with the message."""
    usage_metadata: Optional[UsageMetadata] = None
    """If provided, usage metadata for a message, such as token counts.

    This is a standard representation of token usage that is consistent across models.
    """

    type: Literal["ai"] = "ai"
    """The type of the message (used for deserialization). Defaults to "ai"."""

    def __init__(
        self, content: Union[str, List[Union[str, Dict]]], **kwargs: Any
    ) -> None:
        """Pass in content as positional arg.

        Args:
            content: The content of the message.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(content=content, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.

        Returns:
            The namespace of the langchain object.
            Defaults to ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @root_validator(pre=True)
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
        """Return a pretty representation of the message.

        Args:
            html: Whether to return an HTML-formatted string.
                 Defaults to False.

        Returns:
            A pretty representation of the message.
        """
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
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore
    """The type of the message (used for deserialization). 
    Defaults to "AIMessageChunk"."""

    tool_call_chunks: List[ToolCallChunk] = []
    """If provided, tool call chunks associated with the message."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.

        Returns:
            The namespace of the langchain object.
            Defaults to ["langchain", "schema", "messages"].
        """
        return ["langchain", "schema", "messages"]

    @property
    def lc_attributes(self) -> Dict:
        """Attrs to be serialized even if they are derived from other init args."""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @root_validator(pre=False, skip_on_failure=True)
    def init_tool_calls(cls, values: dict) -> dict:
        """Initialize tool calls from tool call chunks.

        Args:
            values: The values to validate.

        Returns:
            The values with tool calls initialized.

        Raises:
            ValueError: If the tool call chunks are malformed.
        """
        if not values["tool_call_chunks"]:
            if values["tool_calls"]:
                values["tool_call_chunks"] = [
                    ToolCallChunk(
                        name=tc["name"],
                        args=json.dumps(tc["args"]),
                        id=tc["id"],
                        index=None,
                    )
                    for tc in values["tool_calls"]
                ]
            if values["invalid_tool_calls"]:
                tool_call_chunks = values.get("tool_call_chunks", [])
                tool_call_chunks.extend(
                    [
                        ToolCallChunk(
                            name=tc["name"], args=tc["args"], id=tc["id"], index=None
                        )
                        for tc in values["invalid_tool_calls"]
                    ]
                )
                values["tool_call_chunks"] = tool_call_chunks

            return values
        tool_calls = []
        invalid_tool_calls = []
        for chunk in values["tool_call_chunks"]:
            try:
                args_ = parse_partial_json(chunk["args"]) if chunk["args"] != "" else {}
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
            return add_ai_message_chunks(self, other)
        elif isinstance(other, (list, tuple)) and all(
            isinstance(o, AIMessageChunk) for o in other
        ):
            return add_ai_message_chunks(self, *other)
        return super().__add__(other)


def add_ai_message_chunks(
    left: AIMessageChunk, *others: AIMessageChunk
) -> AIMessageChunk:
    """Add multiple AIMessageChunks together."""
    if any(left.example != o.example for o in others):
        raise ValueError(
            "Cannot concatenate AIMessageChunks with different example values."
        )

    content = merge_content(left.content, *(o.content for o in others))
    additional_kwargs = merge_dicts(
        left.additional_kwargs, *(o.additional_kwargs for o in others)
    )
    response_metadata = merge_dicts(
        left.response_metadata, *(o.response_metadata for o in others)
    )

    # Merge tool call chunks
    if raw_tool_calls := merge_lists(
        left.tool_call_chunks, *(o.tool_call_chunks for o in others)
    ):
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

    # Token usage
    if left.usage_metadata or any(o.usage_metadata is not None for o in others):
        usage_metadata_: UsageMetadata = left.usage_metadata or UsageMetadata(
            input_tokens=0, output_tokens=0, total_tokens=0
        )
        for other in others:
            if other.usage_metadata is not None:
                usage_metadata_["input_tokens"] += other.usage_metadata["input_tokens"]
                usage_metadata_["output_tokens"] += other.usage_metadata[
                    "output_tokens"
                ]
                usage_metadata_["total_tokens"] += other.usage_metadata["total_tokens"]
        usage_metadata: Optional[UsageMetadata] = usage_metadata_
    else:
        usage_metadata = None

    return left.__class__(
        example=left.example,
        content=content,
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        id=left.id,
    )
