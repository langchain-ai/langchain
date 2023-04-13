"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence

        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        response = self.queries[list(self.queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response
