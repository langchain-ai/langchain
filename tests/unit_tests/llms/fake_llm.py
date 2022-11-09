"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    def __init__(self, queries: Optional[Mapping] = None):
        """Initialize with optional lookup of queries."""
        self._queries = queries

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self._queries is not None:
            return self._queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
