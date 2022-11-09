"""Base interface for large language models to expose."""
from abc import ABC, abstractmethod
from typing import List, Optional, Mapping, Any


class LLM(ABC):
    """LLM wrapper should take in a prompt and return a string."""

    @abstractmethod
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""

    @property
    @abstractmethod
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Cohere API."""

    def __str__(self):
        """Get a string representation of the object for printing."""
        return f"{self.__class__.__name__}\nParams: {self._default_params}"
