import warnings
from typing import Any

from langchain_core.memory import BaseMemory
from pydantic import field_validator

from langchain.memory.chat_memory import BaseChatMemory


class CombinedMemory(BaseMemory):
    """Combining multiple memories' data together."""

    memories: list[BaseMemory]
    """For tracking all the memories that should be accessed."""

    @field_validator("memories")
    @classmethod
    def check_repeated_memory_variable(
        cls,
        value: list[BaseMemory],
    ) -> list[BaseMemory]:
        all_variables: set[str] = set()
        for val in value:
            overlap = all_variables.intersection(val.memory_variables)
            if overlap:
                msg = (
                    f"The same variables {overlap} are found in multiple"
                    "memory object, which is not allowed by CombinedMemory."
                )
                raise ValueError(msg)
            all_variables |= set(val.memory_variables)

        return value

    @field_validator("memories")
    @classmethod
    def check_input_key(cls, value: list[BaseMemory]) -> list[BaseMemory]:
        """Check that if memories are of type BaseChatMemory that input keys exist."""
        for val in value:
            if isinstance(val, BaseChatMemory) and val.input_key is None:
                warnings.warn(
                    "When using CombinedMemory, "
                    "input keys should be so the input is known. "
                    f" Was not set on {val}",
                    stacklevel=5,
                )
        return value

    @property
    def memory_variables(self) -> list[str]:
        """All the memory variables that this instance provides."""
        """Collected from the all the linked memories."""

        memory_variables = []

        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)

        return memory_variables

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load all vars from sub-memories."""
        memory_data: dict[str, Any] = {}

        # Collect vars from all sub-memories
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            for key, value in data.items():
                if key in memory_data:
                    msg = f"The variable {key} is repeated in the CombinedMemory."
                    raise ValueError(msg)
                memory_data[key] = value

        return memory_data

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save context from this session for every memory."""
        # Save context for all sub-memories
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear context from this session for every memory."""
        for memory in self.memories:
            memory.clear()
