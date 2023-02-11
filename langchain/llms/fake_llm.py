"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from langchain.llms.base import LLM


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    n: int = 1

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None or len(prompt) > 10000:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
