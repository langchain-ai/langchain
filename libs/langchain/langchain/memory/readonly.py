from typing import Any

from langchain_core.memory import BaseMemory


class ReadOnlySharedMemory(BaseMemory):
    """Memory wrapper that is read-only and cannot be changed."""

    memory: BaseMemory

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables."""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load memory variables from memory."""
        return self.memory.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Nothing should be saved or changed"""

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
