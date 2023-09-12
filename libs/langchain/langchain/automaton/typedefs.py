from __future__ import annotations

import abc
from typing import Any, Iterator, Mapping, Optional, Sequence, Union, List

from langchain.load.serializable import Serializable
from langchain.schema import BaseMessage, Document
from langchain.schema.runnable import RunnableConfig


class InternalMessage(Serializable):
    @property
    def lc_serializable(self) -> bool:
        """Indicate whether the class is serializable."""
        return True


class FunctionCallRequest(
    InternalMessage
):  # TODO(Eugene): Rename as FunctionCallRequest
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

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"FunctionCall(name={self.name}, named_arguments={self.named_arguments})"


class FunctionCallResponse(InternalMessage):  # Rename as FunctionCallResult
    """A result of a function invocation."""

    name: str
    result: Any
    error: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"FunctionResult(name={self.name}, result={self.result}, "
            f"error={self.error})"
        )


class RetrievalRequest(InternalMessage):
    """A request for a retrieval."""

    query: str
    """The query to use for the retrieval."""

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"RetrievalRequest(query={self.query})"


class RetrievalResponse(InternalMessage):
    """A result of a retrieval."""

    results: Sequence[Document]

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"RetrievalResults(results={self.results})"


class AdHocMessage(InternalMessage):
    """A message that is used to prime the language model."""

    type: str
    data: Any  # Make sure this is serializable

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"AdHocMessage(type={self.type}, data={self.data})"


class AgentFinish(InternalMessage):
    """A message that indicates that the agent is finished."""

    result: Any

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"AgentFinish(result={self.result})"


MessageLike = Union[BaseMessage, InternalMessage]


class Agent(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        messages: Sequence[MessageLike],
        *,
        config: Optional[dict] = None,
        max_iterations: int = 100,
    ) -> Iterator[MessageLike]:
        """Run the agent."""
        raise NotImplementedError()
