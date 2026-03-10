"""Unit tests for ChatMistralAI retry and max-concurrency handling."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from langchain_mistralai.chat_models import (
    ChatMistralAI,
    _RetryableHTTPStatusError,
    _araise_retryable_status_error,
    _is_retryable_status_error,
    _raise_retryable_status_error,
)

os.environ["MISTRAL_API_KEY"] = "test-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=MagicMock(), response=response
    )


def _make_mock_response(status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.url = "https://api.mistral.ai/v1/chat/completions"
    response.request = MagicMock()
    response.read.return_value = b"error body"
    return response


# ---------------------------------------------------------------------------
# _is_retryable_status_error
# ---------------------------------------------------------------------------

class TestIsRetryableStatusError:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_codes(self, code: int) -> None:
        exc = _make_http_status_error(code)
        assert _is_retryable_status_error(exc) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_non_retryable_codes(self, code: int) -> None:
        exc = _make_http_status_error(code)
        assert _is_retryable_status_error(exc) is False

    def test_non_http_exception_not_retryable(self) -> None:
        assert _is_retryable_status_error(ValueError("bad")) is False

    def test_request_error_not_matched(self) -> None:
        exc = httpx.RequestError("conn error", request=MagicMock())
        assert _is_retryable_status_error(exc) is False


# ---------------------------------------------------------------------------
# _raise_retryable_status_error (sync)
# ---------------------------------------------------------------------------

class TestRaiseRetryableStatusError:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_code_raises_retryable_error(self, code: int) -> None:
        response = _make_mock_response(code)
        with pytest.raises(_RetryableHTTPStatusError):
            _raise_retryable_status_error(response)

    @pytest.mark.parametrize("code", [400, 401, 404])
    def test_non_retryable_code_raises_plain_http_error(self, code: int) -> None:
        response = _make_mock_response(code)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            _raise_retryable_status_error(response)
        # Must NOT be the retryable subclass
        assert type(exc_info.value) is httpx.HTTPStatusError

    def test_success_response_does_not_raise(self) -> None:
        response = _make_mock_response(200)
        _raise_retryable_status_error(response)  # should not raise


# ---------------------------------------------------------------------------
# _araise_retryable_status_error (async)
# ---------------------------------------------------------------------------

class TestARaiseRetryableStatusError:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    async def test_retryable_code_raises_retryable_error(self, code: int) -> None:
        response = _make_mock_response(code)
        response.aread = AsyncMock(return_value=b"error")
        with pytest.raises(_RetryableHTTPStatusError):
            await _araise_retryable_status_error(response)

    @pytest.mark.asyncio
    async def test_success_response_does_not_raise(self) -> None:
        response = _make_mock_response(200)
        await _araise_retryable_status_error(response)  # should not raise


# ---------------------------------------------------------------------------
# ChatMistralAI – retry via completion_with_retry (sync)
# ---------------------------------------------------------------------------

class TestChatMistralAIRetrySync:
    def test_retry_on_request_error_then_success(self) -> None:
        """Tenacity retries httpx.RequestError and succeeds on second attempt."""
        chat = ChatMistralAI(max_retries=3)
        call_count = 0

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.RequestError("connection reset", request=MagicMock())
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            return response

        with patch.object(chat.client, "post", side_effect=mock_post):
            result = chat.invoke("Hello")
        assert result.content == "ok"
        assert call_count == 2

    def test_retry_on_503_then_success(self) -> None:
        """_RetryableHTTPStatusError (503) is retried by tenacity."""
        chat = ChatMistralAI(max_retries=3)
        call_count = 0

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                response = _make_mock_response(503)
                return response  # _raise_retryable_status_error is called inside
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            return response

        with patch.object(chat.client, "post", side_effect=mock_post):
            result = chat.invoke("Hello")
        assert result.content == "ok"
        assert call_count == 3

    def test_non_retryable_404_surfaces_immediately(self) -> None:
        """A 404 (non-retryable) is not retried and raises immediately."""
        chat = ChatMistralAI(max_retries=3)
        call_count = 0

        def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _make_mock_response(404)

        with patch.object(chat.client, "post", side_effect=mock_post):
            with pytest.raises(httpx.HTTPStatusError):
                chat.invoke("Hello")
        assert call_count == 1

    def test_max_retries_respected(self) -> None:
        """After max_retries exhausted, the last exception propagates."""
        chat = ChatMistralAI(max_retries=2)
        call_count = 0

        def mock_post(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise httpx.RequestError("always fails", request=MagicMock())

        with patch.object(chat.client, "post", side_effect=mock_post):
            with pytest.raises(Exception):
                chat.invoke("Hello")
        # 1 initial attempt + 2 retries = 3 total
        assert call_count == 3


# ---------------------------------------------------------------------------
# ChatMistralAI – max_concurrent_requests semaphore (async)
# ---------------------------------------------------------------------------

class TestChatMistralAIMaxConcurrency:
    def test_semaphore_created_with_correct_limit(self) -> None:
        chat = ChatMistralAI(max_concurrent_requests=5)
        assert isinstance(chat._concurrency_semaphore, asyncio.Semaphore)
        # Semaphore._value is the initial count
        assert chat._concurrency_semaphore._value == 5  # type: ignore[attr-defined]

    def test_default_max_concurrent_requests(self) -> None:
        chat = ChatMistralAI()
        assert chat.max_concurrent_requests == 64
        assert chat._concurrency_semaphore._value == 64  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_calls(self) -> None:
        """At most max_concurrent_requests coroutines hold the semaphore at once."""
        max_concurrency = 3
        chat = ChatMistralAI(max_concurrent_requests=max_concurrency, max_retries=0)

        active: list[int] = []
        peak = [0]

        async def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            active.append(1)
            peak[0] = max(peak[0], sum(active))
            await asyncio.sleep(0.05)
            active.pop()
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.aread = AsyncMock(return_value=b"")
            response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            return response

        with patch.object(chat.async_client, "post", side_effect=mock_post):
            tasks = [
                chat.ainvoke("Hello")
                for _ in range(max_concurrency * 3)
            ]
            await asyncio.gather(*tasks)

        assert peak[0] <= max_concurrency

    @pytest.mark.asyncio
    async def test_semaphore_limit_one_is_serial(self) -> None:
        """max_concurrent_requests=1 forces strict serial async execution."""
        chat = ChatMistralAI(max_concurrent_requests=1, max_retries=0)
        active: list[int] = []
        peak = [0]

        async def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            active.append(1)
            peak[0] = max(peak[0], sum(active))
            await asyncio.sleep(0.02)
            active.pop()
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.aread = AsyncMock(return_value=b"")
            response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            return response

        with patch.object(chat.async_client, "post", side_effect=mock_post):
            await asyncio.gather(*[chat.ainvoke("Hello") for _ in range(5)])

        assert peak[0] == 1
