import os
from typing import cast
from unittest.mock import MagicMock

import httpx
from pydantic import SecretStr

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.embeddings import (
    DummyTokenizer,
    _is_retryable_error,
)

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistral_init() -> None:
    for model in [
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key="test"),  # type: ignore[call-arg]
        MistralAIEmbeddings(model="mistral-embed", api_key="test"),  # type: ignore[arg-type]
    ]:
        assert model.model == "mistral-embed"
        assert cast("SecretStr", model.mistral_api_key).get_secret_value() == "test"


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


def test_dummy_tokenizer() -> None:
    """Test that DummyTokenizer returns character lists."""
    tokenizer = DummyTokenizer()
    result = tokenizer.encode_batch(["hello", "world"])
    assert result == [["h", "e", "l", "l", "o"], ["w", "o", "r", "l", "d"]]
