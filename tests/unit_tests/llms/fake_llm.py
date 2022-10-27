"""Fake LLM wrapper for testing purposes."""
from typing import List, Mapping, Optional

from langchain.llms.base import LLM, CompletionOutput


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    def __init__(self, queries: Optional[Mapping] = None):
        """Initialize with optional lookup of queries."""
        self._queries = queries

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> List[CompletionOutput]:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self._queries is not None:
            return [CompletionOutput(text=self._queries[prompt])]
        if stop is None:
            return [CompletionOutput(text="foo")]
        else:
            return [CompletionOutput(text="bar")]
