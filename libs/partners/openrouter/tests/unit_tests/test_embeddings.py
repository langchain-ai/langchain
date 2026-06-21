"""Unit tests for OpenRouterEmbeddings."""

import os
from typing import cast
from unittest.mock import MagicMock

import httpx
from pydantic import SecretStr

from langchain_openrouter import OpenRouterEmbeddings
from langchain_openrouter.embeddings import (
    _is_retryable_error,
)

os.environ["OPENROUTER_API_KEY"] = "foo"


def test_openrouter_init() -> None:
    """Test OpenRouterEmbeddings initialization with different parameter styles."""
    for model in [
        OpenRouterEmbeddings(
            model="openai/text-embedding-3-small", openrouter_api_key="test"
        ),  # type: ignore[call-arg]
        OpenRouterEmbeddings(model="openai/text-embedding-3-small", api_key="test"),  # type: ignore[arg-type]
    ]:
        assert model.model == "openai/text-embedding-3-small"
        assert cast("SecretStr", model.openrouter_api_key).get_secret_value() == "test"


def test_is_retryable_error_timeout() -> None:
    """Test that timeout exceptions are retryable."""
    exc = httpx.TimeoutException("timeout")
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_rate_limit() -> None:
    """Test that 429 errors are retryable."""
    response = MagicMock()
    response.status_code = 429
    exc = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_server_error() -> None:
    """Test that 5xx errors are retryable."""
    for status_code in [500, 502, 503, 504]:
        response = MagicMock()
        response.status_code = status_code
        exc = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=response
        )
        assert _is_retryable_error(exc) is True


def test_is_retryable_error_bad_request_not_retryable() -> None:
    """Test that 400 errors are NOT retryable."""
    response = MagicMock()
    response.status_code = 400
    exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is False


def test_is_retryable_error_other_4xx_not_retryable() -> None:
    """Test that other 4xx errors are NOT retryable."""
    for status_code in [401, 403, 404, 422]:
        response = MagicMock()
        response.status_code = status_code
        exc = httpx.HTTPStatusError(
            "client error", request=MagicMock(), response=response
        )
        assert _is_retryable_error(exc) is False


def test_is_retryable_error_other_exceptions() -> None:
    """Test that other exceptions are not retryable."""
    assert _is_retryable_error(ValueError("test")) is False
    assert _is_retryable_error(RuntimeError("test")) is False
