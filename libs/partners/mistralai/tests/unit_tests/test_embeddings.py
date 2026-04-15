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


def test_max_concurrent_requests_wired_to_httpx_limits() -> None:
    """Verify max_concurrent_requests is forwarded to the httpx connection pool."""
    embed = MistralAIEmbeddings(max_concurrent_requests=4)  # type: ignore[call-arg]
    # httpx.Client does not expose .limits publicly; the chosen value is
    # visible on the underlying httpcore connection pool.
    assert embed.client._transport._pool._max_connections == 4
    assert embed.client._transport._pool._max_keepalive_connections == 4
    assert embed.async_client._transport._pool._max_connections == 4
    assert embed.async_client._transport._pool._max_keepalive_connections == 4


def test_aembed_documents_respects_max_concurrency() -> None:
    """Semaphore in aembed_documents must cap concurrent HTTP calls."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    # Observe the peak concurrency during parallel batch calls.
    peak_concurrency: list[int] = [0]
    active: list[int] = [0]

    async def _fake_post(*args: object, **kwargs: object) -> MagicMock:
        active[0] += 1
        peak_concurrency[0] = max(peak_concurrency[0], active[0])
        await asyncio.sleep(0)  # yield to event loop
        active[0] -= 1

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        return response

    embed = MistralAIEmbeddings(max_concurrent_requests=2, max_retries=None)  # type: ignore[call-arg]

    # Replace DummyTokenizer so each text is its own batch (1 token each).
    embed.tokenizer = DummyTokenizer()
    # Patch the async client post method
    embed.async_client.post = AsyncMock(side_effect=_fake_post)  # type: ignore[method-assign]

    # 5 texts => 5 batches (each 1 char via DummyTokenizer, well under 16k limit)
    texts = ["a", "b", "c", "d", "e"]
    asyncio.run(embed.aembed_documents(texts))

    # Peak concurrency must not exceed max_concurrent_requests.
    assert peak_concurrency[0] <= 2, (
        f"Expected max 2 concurrent requests but saw {peak_concurrency[0]}"
    )
