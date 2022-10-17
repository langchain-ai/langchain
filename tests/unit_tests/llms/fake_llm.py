"""Fake LLM wrapper for testing purposes."""
from typing import List, Optional

from langchain.llms.base import LLM


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Return `foo` if no stop words, otherwise `bar`."""
        if stop is None:
            return "foo"
        else:
            return "bar"
