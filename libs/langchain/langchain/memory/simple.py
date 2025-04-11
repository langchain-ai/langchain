from typing import Any

from langchain_core.memory import BaseMemory


class SimpleMemory(BaseMemory):
    """Simple memory for storing context or other information that shouldn't
    ever change between prompts.
    """

    memories: dict[str, Any] = dict()

    @property
    def memory_variables(self) -> list[str]:
        return list(self.memories.keys())

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return self.memories

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Nothing should be saved or changed, my memory is set in stone."""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
