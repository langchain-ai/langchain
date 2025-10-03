"""Function Message."""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class FunctionMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model.

    ``FunctionMessage`` are an older version of the ``ToolMessage`` schema, and
    do not contain the ``tool_call_id`` field.

    The ``tool_call_id`` field is used to associate the tool call request with the
    tool call response. This is useful in situations where a chat model is able
    to request multiple tool calls in parallel.

    """

    name: str
    """The name of the function that was executed."""

    type: Literal["function"] = "function"
    """The type of the message (used for serialization). Defaults to ``'function'``."""


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """Function Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).

    Defaults to ``'FunctionMessageChunk'``.

    """

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                msg = "Cannot concatenate FunctionMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                name=self.name,
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
