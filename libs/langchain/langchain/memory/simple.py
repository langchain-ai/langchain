from typing import Any, Dict, List

from langchain_core.memory import BaseMemory


class SimpleMemory(BaseMemory):
    """Simple memory for storing context or other information that shouldn't
    ever change between prompts.
    """

    memories: Dict[str, Any] = dict()

    @property
    def memory_variables(self) -> List[str]:
        return list(self.memories.keys())

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.memories

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed, my memory is set in stone."""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
