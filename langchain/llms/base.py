"""Base interface for large language models to expose."""
from abc import ABC, abstractmethod
from typing import List, Optional


class LLM(ABC):
    """LLM wrapper should take in a prompt and return a string."""

    @abstractmethod
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
