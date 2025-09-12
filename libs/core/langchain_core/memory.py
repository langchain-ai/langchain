"""**Memory** maintains Chain state, incorporating context from past runs.

This module contains memory abstractions from LangChain v0.0.x.

These abstractions are now deprecated and will be removed in LangChain v1.0.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ConfigDict

from langchain_core._api import deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import run_in_executor


@deprecated(
    since="0.3.3",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseMemory(Serializable, ABC):
    """Abstract base class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: dict[str, Any] = dict()

                @property
                def memory_variables(self) -> list[str]:
                    return list(self.memories.keys())

                def load_memory_variables(
                    self, inputs: dict[str, Any]
                ) -> dict[str, str]:
                    return self.memories

                def save_context(
                    self, inputs: dict[str, Any], outputs: dict[str, str]
                ) -> None:
                    pass

                def clear(self) -> None:
                    pass

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        Args:
            inputs: The inputs to the chain.

        Returns:
            A dictionary of key-value pairs.
        """

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async return key-value pairs given the text input to the chain.

        Args:
            inputs: The inputs to the chain.

        Returns:
            A dictionary of key-value pairs.
        """
        return await run_in_executor(None, self.load_memory_variables, inputs)

    @abstractmethod
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the context of this chain run to memory.

        Args:
            inputs: The inputs to the chain.
            outputs: The outputs of the chain.
        """

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """Async save the context of this chain run to memory.

        Args:
            inputs: The inputs to the chain.
            outputs: The outputs of the chain.
        """
        await run_in_executor(None, self.save_context, inputs, outputs)

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""

    async def aclear(self) -> None:
        """Async clear memory contents."""
        await run_in_executor(None, self.clear)
