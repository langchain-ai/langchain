from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, Mapping, overload, Union

from langchain.schema import (
    BaseMessage,
)


@dataclasses.dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: Optional[Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class FunctionResult:
    name: str
    result: Any
    error: Optional[str]


@dataclasses.dataclass(frozen=True)
class AgentFinish:
    result: Any


@dataclasses.dataclass(frozen=True)
class PrimingMessage:
    """A message that is used to prime the language model."""

    content: str


MessageLike = Union[
    BaseMessage, FunctionCall, FunctionResult, AgentFinish, PrimingMessage
]


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


class Agent:
    def run(self, message_log: MessageLog) -> None:
        """Run the agent on a message."""
        raise NotImplementedError
