"""Messages for tools."""

import json
from typing import Any, Literal, cast, overload
from uuid import UUID

from pydantic import Field, model_validator
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content
from langchain_core.messages.content import InvalidToolCall
from langchain_core.utils._merge import merge_dicts, merge_obj


class ToolOutputMixin:
    """Mixin for objects that tools can return directly.

    If a custom BaseTool is invoked with a `ToolCall` and the output of custom code is
    not an instance of `ToolOutputMixin`, the output will automatically be coerced to
    a string and wrapped in a `ToolMessage`.

    """


class ToolMessage(BaseMessage, ToolOutputMixin):
    """Message for passing the result of executing a tool back to a model.

    `ToolMessage` objects contain the result of a tool invocation. Typically, the result
    is encoded inside the `content` field.

    `tool_call_id` is used to associate the tool call request with the tool call
    response. Useful in situations where a chat model is able to request multiple tool
    calls in parallel.

    Example:
        A `ToolMessage` representing a result of `42` from a tool call with id

        ```python
        from langchain_core.messages import ToolMessage

        ToolMessage(content="42", tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL")
        ```

    Example:
        A `ToolMessage` where only part of the tool output is sent to the model
        and the full output is passed in to artifact.

        ```python
        from langchain_core.messages import ToolMessage

        tool_output = {
            "stdout": "From the graph we can see that the correlation between "
            "x and y is ...",
            "stderr": None,
            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
        }

        ToolMessage(
            content=tool_output["stdout"],
            artifact=tool_output,
            tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
        )
        ```
    """

    tool_call_id: str
    """Tool call that this message is responding to."""

    type: Literal["tool"] = "tool"
    """The type of the message (used for serialization)."""

    artifact: Any = None
    """Artifact of the Tool execution which is not meant to be sent to the model.

    Should only be specified if it is different from the message content, e.g. if only
    a subset of the full tool output is being passed as message content but the full
    output is needed in other parts of the code.

    """

    status: Literal["success", "error"] = "success"
    """Status of the tool invocation."""

    additional_kwargs: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from `BaseMessage`, but not used."""
    response_metadata: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from `BaseMessage`, but not used."""

    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values: dict) -> dict:
        """Coerce the model arguments to the correct types.

        Args:
            values: The model arguments.

        """
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

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

    @overload
    def __init__(
        self,
        content: str | list[str | dict],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `ToolMessage`.

        Specify `content` as positional arg or `content_blocks` for typing.

        Args:
            content: The contents of the message.
            content_blocks: Typed standard content.
            **kwargs: Additional fields.
        """
        if content_blocks is not None:
            super().__init__(
                content=cast("str | list[str | dict]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
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
    """Represents an AI's request to call a tool.

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named `'foo'` with arguments
        `{"a": 1}` and an identifier of `'123'`.

    !!! note "Factory function"

        `tool_call` may also be used as a factory to create a `ToolCall`. Benefits
        include:

        * Required arguments strictly validated at creation time
    """

    name: str
    """The name of the tool to be called."""

    args: dict[str, Any]
    """The arguments to the tool call as a dictionary."""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """

    type: NotRequired[Literal["tool_call"]]
    """Used for discrimination."""


def tool_call(
    *,
    name: str,
    args: dict[str, Any],
    id: str | None,
) -> ToolCall:
    """Create a tool call.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call as a dictionary.
        id: An identifier associated with the tool call.

    Returns:
        The created tool call.
    """
    return ToolCall(name=name, args=args, id=id, type="tool_call")


class ToolCallChunk(TypedDict):
    """A chunk of a tool call (yielded when streaming).

    When merging `ToolCallChunk` objects (e.g., via `AIMessageChunk.__add__`), all
    string attributes are concatenated. Chunks are only merged if their values of
    `index` are equal and not `None`.

    Example:
    ```python
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]

    (
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ```
    """

    name: str | None
    """The name of the tool to be called."""

    args: str | None
    """The arguments to the tool call as a JSON-parseable string."""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """

    index: int | None
    """The index of the tool call in a sequence.

    Used for merging chunks.
    """

    type: NotRequired[Literal["tool_call_chunk"]]
    """Used for discrimination."""


def tool_call_chunk(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    index: int | None = None,
) -> ToolCallChunk:
    """Create a tool call chunk.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call as a JSON string.
        id: An identifier associated with the tool call.
        index: The index of the tool call in a sequence.

    Returns:
        The created tool call chunk.
    """
    return ToolCallChunk(
        name=name, args=args, id=id, index=index, type="tool_call_chunk"
    )


def invalid_tool_call(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    error: str | None = None,
) -> InvalidToolCall:
    """Create an invalid tool call.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call as a JSON string.
        id: An identifier associated with the tool call.
        error: An error message associated with the tool call.

    Returns:
        The created invalid tool call.
    """
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


def default_tool_parser(
    raw_tool_calls: list[dict],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """Best-effort parsing of tools.

    Args:
        raw_tool_calls: List of raw tool call dicts to parse.

    Returns:
        A list of tool calls and invalid tool calls.
    """
    tool_calls = []
    invalid_tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
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
    """Best-effort parsing of tool chunks.

    Args:
        raw_tool_calls: List of raw tool call dicts to parse.

    Returns:
        List of parsed ToolCallChunk objects.
    """
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
    return "error" if "error" in {left, right} else "success"
