from abc import ABC, abstractmethod
from typing import Optional, List

import langchain
from langchain.llms import LLM


class StreamingLLM(LLM, ABC):
    @abstractmethod
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)
