"""Shared fixtures for TypeSafe middleware unit tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import httpx2
import pytest
import typesafe_sdk as ts
from langgraph.runtime import Runtime

API_KEY = "test-api-key"

RUNTIME: Runtime[Any] = cast("Runtime[Any]", None)
"""The middleware under test ignore `runtime`, so a stand-in keeps call sites typed."""

NO_RETRIES = ts.RetryPolicy(max_retries=0)

Handler = Callable[[httpx2.Request], httpx2.Response]


def answers_response(answers: dict[str, Any]) -> dict[str, Any]:
    """Build a `POST /v1/systemone` response body with the given answers."""
    return {
        "model": "jev-latest",
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


class RecordingTransport:
    """Mock transport that records request bodies and replays fixed responses."""

    def __init__(self, response: dict[str, Any] | int) -> None:
        """Store the response to replay for every request."""
        self.requests: list[dict[str, Any]] = []
        self._response = response

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Record the request body and return the configured response."""
        self.requests.append(json.loads(request.content))
        if isinstance(self._response, int):
            return httpx2.Response(self._response)
        return httpx2.Response(200, json=self._response)

    @property
    def state(self) -> Any:
        """Return the `state` field of the single recorded request."""
        assert len(self.requests) == 1
        return self.requests[0]["state"]

    @property
    def questions(self) -> dict[str, Any]:
        """Return the `questions` field of the single recorded request."""
        assert len(self.requests) == 1
        return self.requests[0]["questions"]


@pytest.fixture
def clients() -> Callable[[Handler], dict[str, Any]]:
    """Return a factory building middleware client kwargs from a handler."""

    def build(handler: Handler) -> dict[str, Any]:
        return {
            "api_key": API_KEY,
            "client": ts.TypeSafeClient(
                api_key=API_KEY,
                transport=httpx2.MockTransport(handler),
                retry=NO_RETRIES,
            ),
            "async_client": ts.AsyncTypeSafeClient(
                api_key=API_KEY,
                transport=httpx2.MockTransport(handler),
                retry=NO_RETRIES,
            ),
        }

    return build
