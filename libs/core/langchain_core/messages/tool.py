import json
from typing import Any, Literal, Optional, Union
from uuid import UUID

from pydantic import Field, model_validator
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content
from langchain_core.utils._merge import merge_dicts, merge_obj


class ToolMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model.

    ToolMessages contain the result of a tool invocation. Typically, the result
    is encoded inside the `content` field.

    Example: A ToolMessage representing a result of 42 from a tool call with id

        .. code-block:: python

            from langchain_core.messages import ToolMessage

            ToolMessage(content='42', tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL')


    Example: A ToolMessage where only part of the tool output is sent to the model
        and the full output is passed in to artifact.

        .. versionadded:: 0.2.17

        .. code-block:: python

            from langchain_core.messages import ToolMessage

            tool_output = {
                "stdout": "From the graph we can see that the correlation between x and y is ...",
                "stderr": None,
                "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
            }

            ToolMessage(
                content=tool_output["stdout"],
                artifact=tool_output,
                tool_call_id='call_Jja7J89XsjrOLA5r!MEOW!SL',
            )

    The tool_call_id field is used to associate the tool call request with the
    tool call response. This is useful in situations where a chat model is able
    to request multiple tool calls in parallel.
    """  # noqa: E501

    tool_call_id: str
    """Tool call that this message is responding to."""

    type: Literal["tool"] = "tool"
    """The type of the message (used for serialization). Defaults to "tool"."""

    artifact: Any = None
    """Artifact of the Tool execution which is not meant to be sent to the model.

    Should only be specified if it is different from the message content, e.g. if only
    a subset of the full tool output is being passed as message content but the full
    output is needed in other parts of the code.

    .. versionadded:: 0.2.17
    """

    status: Literal["success", "error"] = "success"
    """Status of the tool invocation.

    .. versionadded:: 0.2.24
    """

    additional_kwargs: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from BaseMessage, but not used."""
    response_metadata: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from BaseMessage, but not used."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.
        Default is ["langchain", "schema", "messages"]."""
        return ["langchain", "schema", "messages"]

    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values: dict) -> dict:
        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)

        if not isinstance(content, (str, list)):
            try:
                values["content"] = str(content)
            except ValueError as e:
                msg = (
                    "ToolMessage content should be a string or a list of string/dicts. "
                    f"Received:\n\n{content=}\n\n which could not be coerced into a "
                    "string."
                )
                raise ValueError(msg) from e
        elif isinstance(content, list):
            values["content"] = []
            for i, x in enumerate(content):
                if not isinstance(x, (str, dict)):
                    try:
                        values["content"].append(str(x))
                    except ValueError as e:
                        msg = (
                            "ToolMessage content should be a string or a list of "
                            "string/dicts. Received a list but "
                            f"element ToolMessage.content[{i}] is not a dict and could "
                            f"not be coerced to a string.:\n\n{x}"
                        )
                        raise ValueError(msg) from e
                else:
                    values["content"].append(x)
        else:
            pass

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        super().__init__(content=content, **kwargs)


ToolMessage.model_rebuild()


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "messages"]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                msg = "Cannot concatenate ToolMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                artifact=merge_obj(self.artifact, other.artifact),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
                status=_merge_status(self.status, other.status),
            )

        return super().__add__(other)


class ToolCall(TypedDict):
    """Represents a request to call a tool.

    Example:

        .. code-block:: python

            {
                "name": "foo",
                "args": {"a": 1},
                "id": "123"
            }

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".
    """

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: Optional[str]
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
    type: NotRequired[Literal["tool_call"]]


def tool_call(*, name: str, args: dict[str, Any], id: Optional[str]) -> ToolCall:
    return ToolCall(name=name, args=args, id=id, type="tool_call")


class ToolCallChunk(TypedDict):
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
    """

    name: Optional[str]
    """The name of the tool to be called."""
    args: Optional[str]
    """The arguments to the tool call."""
    id: Optional[str]
    """An identifier associated with the tool call."""
    index: Optional[int]
    """The index of the tool call in a sequence."""
    type: NotRequired[Literal["tool_call_chunk"]]


def tool_call_chunk(
    *,
    name: Optional[str] = None,
    args: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> ToolCallChunk:
    return ToolCallChunk(
        name=name, args=args, id=id, index=index, type="tool_call_chunk"
    )


class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    """

    name: Optional[str]
    """The name of the tool to be called."""
    args: Optional[str]
    """The arguments to the tool call."""
    id: Optional[str]
    """An identifier associated with the tool call."""
    error: Optional[str]
    """An error message associated with the tool call."""
    type: NotRequired[Literal["invalid_tool_call"]]


def invalid_tool_call(
    *,
    name: Optional[str] = None,
    args: Optional[str] = None,
    id: Optional[str] = None,
    error: Optional[str] = None,
) -> InvalidToolCall:
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


def default_tool_parser(
    raw_tool_calls: list[dict],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """Best-effort parsing of tools."""
    tool_calls = []
    invalid_tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
        else:
            function_name = raw_tool_call["function"]["name"]
            try:
                function_args = json.loads(raw_tool_call["function"]["arguments"])
                parsed = tool_call(
                    name=function_name or "",
                    args=function_args or {},
                    id=raw_tool_call.get("id"),
                )
                tool_calls.append(parsed)
            except json.JSONDecodeError:
                invalid_tool_calls.append(
                    invalid_tool_call(
                        name=function_name,
                        args=raw_tool_call["function"]["arguments"],
                        id=raw_tool_call.get("id"),
                        error=None,
                    )
                )
    return tool_calls, invalid_tool_calls


def default_tool_chunk_parser(raw_tool_calls: list[dict]) -> list[ToolCallChunk]:
    """Best-effort parsing of tool chunks."""
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        parsed = tool_call_chunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks


def _merge_status(
    left: Literal["success", "error"], right: Literal["success", "error"]
) -> Literal["success", "error"]:
    return "error" if "error" in (left, right) else "success"
