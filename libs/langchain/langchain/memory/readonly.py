from typing import Any, Dict, List

from langchain_core.memory import BaseMemory


class ReadOnlySharedMemory(BaseMemory):
    """A memory wrapper that is read-only and cannot be changed."""

    memory: BaseMemory

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory variables from memory."""
        return self.memory.load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
