from __future__ import annotations

from typing import Any, Optional, Sequence, Mapping, overload, Union

from langchain.load.serializable import Serializable
from langchain.schema import (
    BaseMessage,
)


class InternalMessage(Serializable):
    @property
    def lc_serializable(self) -> bool:
        """Indicate whether the class is serializable."""
        return True


class FunctionCall(InternalMessage):
    """A request for a function invocation.

    This message can be used to request a function invocation
    using the function name and the arguments to pass to the function.
    """

    name: str
    """The name of the function to invoke."""
    named_arguments: Optional[Mapping[str, Any]] = None
    """The named arguments to pass to the function."""

    class Config:
        extra = "forbid"

    def __str__(self):
        return f"FunctionCall(name={self.name}, named_arguments={self.named_arguments})"


class FunctionResult(InternalMessage):
    """A result of a function invocation."""

    name: str
    result: Any
    error: Optional[str] = None

    def __str__(self):
        return f"FunctionResult(name={self.name}, result={self.result}, error={self.error})"


class PrimingMessage(InternalMessage):
    """A message that is used to prime the language model."""

    content: str


class AgentFinish(InternalMessage):
    result: Any

    def __str__(self):
        return f"AgentFinish(result={self.result})"


MessageLike = Union[BaseMessage, InternalMessage]


class MessageLog:
    """A generalized message log for message like items."""

    def __init__(self, messages: Sequence[MessageLike] = ()) -> None:
        """Initialize the message log."""
        self.messages = list(messages)

    def add_messages(self, messages: Sequence[MessageLike]) -> None:
        """Add messages to the message log."""
        self.messages.extend(messages)

    @overload
    def __getitem__(self, index: int) -> MessageLike:
        ...

    @overload
    def __getitem__(self, index: slice) -> MessageLog:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[MessageLike, MessageLog]:
        """Use to index into the chat template."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return MessageLog(messages=messages)
        else:
            return self.messages[index]

    def __bool__(self):
        return bool(self.messages)

    def __len__(self) -> int:
        """Get the length of the chat template."""
        return len(self.messages)


class Agent:  # This is just approximate still
    def run(self, message_log: MessageLog) -> None:
        """Run the agent on a message."""
        raise NotImplementedError
