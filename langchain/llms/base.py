"""Base interface for large language models to expose."""
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CompletionOutput:
    """A completion output."""

    text: str 
    """The generated text."""
    logprobs: Optional[List[float]] = None
    """The total log probability assigned to the generated text."""

class LLM(ABC):
    """LLM wrapper that should take in a prompt and return a string."""

    @abstractmethod
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> List[CompletionOutput]:
        """Generate strings for the given prompt and input."""

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""
        return self.generate(prompt=prompt, stop=stop)[0].text
