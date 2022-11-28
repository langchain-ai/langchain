"""Base interface for large language models to expose."""
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional


class LLM(ABC):
    """LLM wrapper should take in a prompt and return a string."""

    @abstractmethod
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"
