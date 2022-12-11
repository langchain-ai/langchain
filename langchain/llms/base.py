"""Base interface for large language models to expose."""
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional, NamedTuple


class LLMResult(NamedTuple):
    """Class that contains all relevant information for an LLM Result."""
    result: str
    log_probs: List[float]
    llm_output: dict
    """For arbitrary LLM provider specific output."""



class LLM(ABC):
    """LLM wrapper should take in a prompt and return a string."""

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> LLMResult:
        """Run the LLM on the given prompt and input."""

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        return self.generate(prompt, stop=stop).result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"
