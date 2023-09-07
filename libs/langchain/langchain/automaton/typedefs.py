from __future__ import annotations

from typing import Any, Iterator, Mapping, Optional, Sequence, Union

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


class AdHocMessage(InternalMessage):
    """A message that is used to prime the language model."""

    type: str
    data: Any  # Make sure this is serializable

    def __str__(self):
        return f"AdHocMessage(type={self.type}, data={self.data})"


class AgentFinish(InternalMessage):
    result: Any

    def __str__(self):
        return f"AgentFinish(result={self.result})"


MessageLike = Union[BaseMessage, InternalMessage]


class Agent:  # This is just approximate still, may end up being a runnable
    def run(self, messages: Sequence[MessageLike]) -> Iterator[MessageLike]:
        """Run the agent on a message."""
        raise NotImplementedError
