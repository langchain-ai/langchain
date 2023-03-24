"""Fake LLM wrapper for testing purposes."""
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from langchain.llms.base import LLM


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes."""

    id = "fake"
    """Unique ID for this provider class."""

    model_id: str = ""
    """
    Model ID to invoke by this provider via generate/agenerate.
    """

    models = ["*"]
    """List of supported models by their IDs. For registry providers, this will
    be just ["*"]."""

    pypi_package_deps = []
    """List of PyPi package dependencies."""

    auth_strategy = None
    """Authentication/authorization strategy. Declares what credentials are
    required to use this model provider. Generally should not be `None`."""

    n: Optional[int] = None

    queries: Optional[Mapping] = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
