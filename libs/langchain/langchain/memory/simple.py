from typing import Any

from langchain_core.memory import BaseMemory
from typing_extensions import override


class SimpleMemory(BaseMemory):
    """Simple memory for storing context or other information that shouldn't
    ever change between prompts.
    """

    memories: dict[str, Any] = {}

    @property
    @override
    def memory_variables(self) -> list[str]:
        return list(self.memories.keys())

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return self.memories

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Nothing should be saved or changed, my memory is set in stone."""

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
