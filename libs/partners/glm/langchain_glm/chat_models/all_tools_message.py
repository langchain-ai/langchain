import json
from typing import Any, Dict, List, Literal, Union

from langchain_core.messages import AIMessage
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


def default_all_tool_chunk_parser(raw_tool_calls: List[dict]) -> List[ToolCallChunk]:
    """Best-effort parsing of all tool chunks."""
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" in tool_call and tool_call["function"] is not None:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        elif (
            "code_interpreter" in tool_call
            and tool_call["code_interpreter"] is not None
        ):
            function_args = json.dumps(
                tool_call["code_interpreter"], ensure_ascii=False
            )
            function_name = "code_interpreter"
        elif "drawing_tool" in tool_call and tool_call["drawing_tool"] is not None:
            function_args = json.dumps(tool_call["drawing_tool"], ensure_ascii=False)
            function_name = "drawing_tool"
        elif "web_browser" in tool_call and tool_call["web_browser"] is not None:
            function_args = json.dumps(tool_call["web_browser"], ensure_ascii=False)
            function_name = "web_browser"
        else:
            function_args = None
            function_name = None
        parsed = ToolCallChunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks


class ALLToolsMessageChunk(AIMessage, BaseMessageChunk):
    """Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ALLToolsMessageChunk"] = "ALLToolsMessageChunk"  # type: ignore[assignment] # noqa: E501

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

    @root_validator(allow_reuse=True)
    def _backwards_compat_tool_calls(cls, values: dict) -> dict:
        raw_tool_calls = values.get("additional_kwargs", {}).get("tool_calls")
        tool_calls = (
            values.get("tool_calls")
            or values.get("invalid_tool_calls")
            or values.get("tool_call_chunks")
        )
        if raw_tool_calls and not tool_calls:
            try:
                if issubclass(cls, BaseMessageChunk):  # type: ignore
                    values["tool_call_chunks"] = default_all_tool_chunk_parser(
                        raw_tool_calls
                    )
                else:
                    tool_calls, invalid_tool_calls = default_tool_parser(raw_tool_calls)
                    values["tool_calls"] = tool_calls
                    values["invalid_tool_calls"] = invalid_tool_calls
            except Exception as e:
                pass
        return values

    @root_validator(allow_reuse=True)
    def init_tool_calls(cls, values: dict) -> dict:
        if not values["tool_call_chunks"]:
            values["tool_calls"] = []
            values["invalid_tool_calls"] = []
            return values
        tool_calls, invalid_tool_calls = _paser_chunk(values["tool_call_chunks"])
        values["tool_calls"] = tool_calls
        values["invalid_tool_calls"] = invalid_tool_calls
        return values

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ALLToolsMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate ALLToolsMessageChunks with different example values."
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


def _paser_chunk(tool_call_chunks):
    tool_calls = []
    invalid_tool_calls = []
    for chunk in tool_call_chunks:
        try:
            if "code_interpreter" in chunk["name"]:
                args_ = parse_partial_json(chunk["args"])

                if not isinstance(args_, dict):
                    raise ValueError("Malformed args.")

                if "outputs" in args_:
                    tool_calls.append(
                        ToolCall(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )

                else:
                    invalid_tool_calls.append(
                        InvalidToolCall(
                            name=chunk["name"],
                            args=chunk["args"],
                            id=chunk["id"],
                            error=None,
                        )
                    )
            elif "drawing_tool" in chunk["name"]:
                args_ = parse_partial_json(chunk["args"])

                if not isinstance(args_, dict):
                    raise ValueError("Malformed args.")

                if "outputs" in args_:
                    tool_calls.append(
                        ToolCall(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )

                else:
                    invalid_tool_calls.append(
                        InvalidToolCall(
                            name=chunk["name"],
                            args=chunk["args"],
                            id=chunk["id"],
                            error=None,
                        )
                    )
            elif "web_browser" in chunk["name"]:
                args_ = parse_partial_json(chunk["args"])

                if not isinstance(args_, dict):
                    raise ValueError("Malformed args.")

                if "outputs" in args_:
                    tool_calls.append(
                        ToolCall(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )

                else:
                    invalid_tool_calls.append(
                        InvalidToolCall(
                            name=chunk["name"],
                            args=chunk["args"],
                            id=chunk["id"],
                            error=None,
                        )
                    )
            else:
                args_ = parse_partial_json(chunk["args"])

                if isinstance(args_, dict):
                    temp_args_ = {}
                    for key, value in args_.items():
                        key = key.strip()
                        temp_args_[key] = value

                    tool_calls.append(
                        ToolCall(
                            name=chunk["name"] or "",
                            args=temp_args_,
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
    return tool_calls, invalid_tool_calls
