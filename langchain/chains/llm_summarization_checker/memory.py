"""Memory for summarization checker chain."""
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.chains.base import Memory


class ImmutableMemory(Memory, BaseModel):
    """Immutable memory for storing context or other bits of information."""

    memories: Dict[str, Any] = dict()

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of it's memory keys."""
        return list(self.memories.keys())

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return the stored memories."""
        return self.memories

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed, my memory is set in stone."""

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
