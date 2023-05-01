from typing import Any, Dict, List

from pydantic import validator

from langchain.schema import BaseMemory


class CombinedMemory(BaseMemory):
    """Class for combining multiple memories' data together."""

    memories: List[BaseMemory]
    """For tracking all the memories that should be accessed."""

    @validator("memories")
    def check_repeated_memory_variable(
        cls, value: List[BaseMemory]
    ) -> List[BaseMemory]:
        for i, cur in enumerate(value):
            for other in value[i + 1 :]:
                repeated_vars = set(cur.memory_variables).intersection(
                    set(other.memory_variables)
                )
                if repeated_vars:
                    raise ValueError(
                        f"{cur} and {other} contain "
                        f"common memory variable {repeated_vars}, "
                        "which is not allowed by CombinedMemory."
                    )

        return value

    @property
    def memory_variables(self) -> List[str]:
        """All the memory variables that this instance provides."""
        """Collected from the all the linked memories."""

        memory_variables = []

        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)

        return memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load all vars from sub-memories."""
        memory_data: Dict[str, Any] = {}

        # Collect vars from all sub-memories
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            memory_data = {
                **memory_data,
                **data,
            }

        return memory_data

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this session for every memory."""
        # Save context for all sub-memories
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear context from this session for every memory."""
        for memory in self.memories:
            memory.clear()
