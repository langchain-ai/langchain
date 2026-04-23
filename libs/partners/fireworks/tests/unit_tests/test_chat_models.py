"""Unit tests for ChatFireworks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fireworks.client.error import (  # type: ignore[import-untyped]
    AuthenticationError,
    FireworksError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from langchain_core.messages import AIMessage, HumanMessage

from langchain_fireworks import ChatFireworks
from langchain_fireworks.chat_models import (
    _acompletion_with_retry,
    _completion_with_retry,
    _convert_dict_to_message,
)


def test_fireworks_model_param() -> None:
    llm = ChatFireworks(model="foo", api_key="fake-key")  # type: ignore[arg-type]
    assert llm.model_name == "foo"
    assert llm.model == "foo"
    llm = ChatFireworks(model_name="foo", api_key="fake-key")  # type: ignore[call-arg, arg-type]
    assert llm.model_name == "foo"
    assert llm.model == "foo"


def test_convert_dict_to_message_with_reasoning_content() -> None:
    """Test that reasoning_content is correctly extracted from API response."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Let me think about this step by step...",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" in message.additional_kwargs
    expected_reasoning = "Let me think about this step by step..."
    assert message.additional_kwargs["reasoning_content"] == expected_reasoning


def test_convert_dict_to_message_without_reasoning_content() -> None:
    """Test that messages without reasoning_content work correctly."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" not in message.additional_kwargs


def _make_llm(max_retries: int | None = 2) -> ChatFireworks:
    return ChatFireworks(
        model="accounts/fireworks/models/test",
        api_key="fake-key",  # type: ignore[arg-type]
        max_retries=max_retries,
    )


def _success_response() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid tenacity's exponential backoff in tests."""
    import asyncio
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)

    async def _no_async_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_async_sleep)


def test_completion_with_retry_retries_on_retryable_error() -> None:
    """Retryable errors trigger retries up to the configured limit."""
    llm = _make_llm(max_retries=2)
    mock_client = MagicMock()
    mock_client.create.side_effect = [
        RateLimitError("rate limited"),
        ServiceUnavailableError("unavailable"),
        _success_response(),
    ]
    llm.client = mock_client

    result = _completion_with_retry(llm, messages=[])

    assert result == _success_response()
    assert mock_client.create.call_count == 3


def test_completion_with_retry_does_not_retry_non_retryable() -> None:
    """Non-retryable errors propagate after a single attempt."""
    llm = _make_llm(max_retries=3)
    mock_client = MagicMock()
    mock_client.create.side_effect = AuthenticationError("bad key")
    llm.client = mock_client

    with pytest.raises(AuthenticationError):
        _completion_with_retry(llm, messages=[])

    assert mock_client.create.call_count == 1


def test_completion_with_retry_respects_max_retries_none() -> None:
    """`max_retries=None` disables retries."""
    llm = _make_llm(max_retries=None)
    mock_client = MagicMock()
    mock_client.create.side_effect = RateLimitError("rate limited")
    llm.client = mock_client

    with pytest.raises(RateLimitError):
        _completion_with_retry(llm, messages=[])

    assert mock_client.create.call_count == 1


def test_completion_with_retry_exhausts_and_raises() -> None:
    """When every attempt fails, the last error is re-raised."""
    llm = _make_llm(max_retries=2)
    mock_client = MagicMock()
    mock_client.create.side_effect = RateLimitError("rate limited")
    llm.client = mock_client

    with pytest.raises(RateLimitError):
        _completion_with_retry(llm, messages=[])

    # 1 initial attempt + 2 retries = 3 total attempts
    assert mock_client.create.call_count == 3


def test_completion_with_retry_streaming_retries_on_setup() -> None:
    """Streaming errors raised during the first-chunk pull are retried."""
    llm = _make_llm(max_retries=1)

    calls = {"n": 0}

    def _fail_then_stream(**_kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:

            def _failing_gen() -> Any:
                msg = "rate limited"
                raise RateLimitError(msg)
                yield  # pragma: no cover

            return _failing_gen()

        def _good_gen() -> Any:
            yield {"id": 0, "choices": [{"delta": {"content": "one"}}]}
            yield {"id": 1, "choices": [{"delta": {"content": "two"}}]}

        return _good_gen()

    mock_client = MagicMock()
    mock_client.create.side_effect = _fail_then_stream
    llm.client = mock_client

    chunks = list(_completion_with_retry(llm, messages=[], stream=True))

    # First chunk is preserved and in order — guards `_prepend_chunk` regression
    assert [c["id"] for c in chunks] == [0, 1]
    assert calls["n"] == 2


def test_completion_with_retry_retries_on_5xx_http_status_error() -> None:
    """5xx `httpx.HTTPStatusError` is promoted and retried."""
    llm = _make_llm(max_retries=1)
    mock_client = MagicMock()
    response_504 = httpx.Response(status_code=504, request=httpx.Request("POST", "x"))
    mock_client.create.side_effect = [
        httpx.HTTPStatusError(
            "504", request=response_504.request, response=response_504
        ),
        _success_response(),
    ]
    llm.client = mock_client

    result = _completion_with_retry(llm, messages=[])

    assert result == _success_response()
    assert mock_client.create.call_count == 2


def test_completion_with_retry_does_not_retry_on_4xx_http_status_error() -> None:
    """Non-5xx `httpx.HTTPStatusError` passes through unretried."""
    llm = _make_llm(max_retries=3)
    mock_client = MagicMock()
    response_422 = httpx.Response(status_code=422, request=httpx.Request("POST", "x"))
    mock_client.create.side_effect = httpx.HTTPStatusError(
        "422", request=response_422.request, response=response_422
    )
    llm.client = mock_client

    with pytest.raises(httpx.HTTPStatusError):
        _completion_with_retry(llm, messages=[])
    assert mock_client.create.call_count == 1


def test_completion_with_retry_retries_on_timeout_exception() -> None:
    """`httpx.TimeoutException` is in the retryable set."""
    llm = _make_llm(max_retries=1)
    mock_client = MagicMock()
    mock_client.create.side_effect = [
        httpx.ConnectTimeout("slow"),
        _success_response(),
    ]
    llm.client = mock_client

    result = _completion_with_retry(llm, messages=[])

    assert result == _success_response()
    assert mock_client.create.call_count == 2


def test_completion_with_retry_max_retries_zero_is_single_attempt() -> None:
    """`max_retries=0` disables retries (same as `None`)."""
    llm = _make_llm(max_retries=0)
    mock_client = MagicMock()
    mock_client.create.side_effect = RateLimitError("rate limited")
    llm.client = mock_client

    with pytest.raises(RateLimitError):
        _completion_with_retry(llm, messages=[])
    assert mock_client.create.call_count == 1


def test_completion_with_retry_raises_on_empty_stream() -> None:
    """Empty streams surface as a descriptive `FireworksError`."""
    llm = _make_llm(max_retries=0)
    mock_client = MagicMock()

    def _empty_gen(**_kwargs: Any) -> Any:
        if False:
            yield  # pragma: no cover
        return

    mock_client.create.side_effect = _empty_gen
    llm.client = mock_client

    with pytest.raises(FireworksError, match="empty stream"):
        list(_completion_with_retry(llm, messages=[], stream=True))


def test_chat_fireworks_invoke_routes_through_retry() -> None:
    """`.invoke()` end-to-end exercises the retry helper on `self.client.create`.

    Guards against a regression that bypasses `_completion_with_retry` from
    `_generate`.
    """
    llm = _make_llm(max_retries=2)
    mock_client = MagicMock()
    mock_client.create.side_effect = [
        RateLimitError("rate limited"),
        _success_response(),
    ]
    llm.client = mock_client

    result = llm.invoke("hello")

    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert mock_client.create.call_count == 2


async def test_acompletion_with_retry_streaming_retries_on_setup() -> None:
    """Async streaming errors during the first-chunk pull are retried."""
    llm = _make_llm(max_retries=1)
    calls = {"n": 0}

    def _acreate(**_kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:

            async def _failing_agen() -> Any:
                msg = "rate limited"
                raise RateLimitError(msg)
                yield  # pragma: no cover

            return _failing_agen()

        async def _good_agen() -> Any:
            yield {"id": 0, "choices": [{"delta": {"content": "one"}}]}
            yield {"id": 1, "choices": [{"delta": {"content": "two"}}]}

        return _good_agen()

    mock_async = MagicMock()
    mock_async.acreate = _acreate
    llm.async_client = mock_async

    agen = await _acompletion_with_retry(llm, messages=[], stream=True)
    chunks = [c async for c in agen]

    assert [c["id"] for c in chunks] == [0, 1]
    assert calls["n"] == 2


async def test_achat_fireworks_ainvoke_routes_through_retry() -> None:
    """`.ainvoke()` end-to-end exercises the async retry helper."""
    llm = _make_llm(max_retries=2)
    calls = {"n": 0}

    async def _acreate(**_kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "rate limited"
            raise RateLimitError(msg)
        return _success_response()

    mock_async = MagicMock()
    mock_async.acreate = _acreate
    llm.async_client = mock_async

    result = await llm.ainvoke("hello")
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert calls["n"] == 2


async def test_acompletion_with_retry_retries_on_retryable_error() -> None:
    """Async retries on retryable errors up to the configured limit."""
    llm = _make_llm(max_retries=2)
    mock_async = MagicMock()

    call_count = {"n": 0}

    async def _acreate(**_kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] < 3:
            msg = "rate limited"
            raise RateLimitError(msg)
        return _success_response()

    mock_async.acreate = _acreate
    llm.async_client = mock_async

    result = await _acompletion_with_retry(llm, messages=[])
    assert result == _success_response()
    assert call_count["n"] == 3


async def test_acompletion_with_retry_does_not_retry_non_retryable() -> None:
    """Async does not retry non-retryable errors."""
    llm = _make_llm(max_retries=3)
    mock_async = MagicMock()
    call_count = {"n": 0}

    async def _acreate(**_kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        msg = "bad input"
        raise InvalidRequestError(msg)

    mock_async.acreate = _acreate
    llm.async_client = mock_async

    with pytest.raises(InvalidRequestError):
        await _acompletion_with_retry(llm, messages=[HumanMessage(content="hi")])
    assert call_count["n"] == 1


async def test_acompletion_with_retry_retries_on_5xx_http_status_error() -> None:
    """Async 5xx `httpx.HTTPStatusError` is promoted and retried."""
    llm = _make_llm(max_retries=1)
    call_count = {"n": 0}
    response_504 = httpx.Response(status_code=504, request=httpx.Request("POST", "x"))

    async def _acreate(**_kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            msg = "504"
            raise httpx.HTTPStatusError(
                msg, request=response_504.request, response=response_504
            )
        return _success_response()

    mock_async = MagicMock()
    mock_async.acreate = _acreate
    llm.async_client = mock_async

    result = await _acompletion_with_retry(llm, messages=[])
    assert result == _success_response()
    assert call_count["n"] == 2


async def test_acompletion_with_retry_raises_on_empty_stream() -> None:
    """Async empty streams surface as a descriptive `FireworksError`."""
    llm = _make_llm(max_retries=0)

    def _acreate(**_kwargs: Any) -> Any:
        async def _empty_agen() -> Any:
            if False:
                yield  # pragma: no cover
            return

        return _empty_agen()

    mock_async = MagicMock()
    mock_async.acreate = _acreate
    llm.async_client = mock_async

    with pytest.raises(FireworksError, match="empty stream"):
        agen = await _acompletion_with_retry(llm, messages=[], stream=True)
        async for _ in agen:
            pass


def test_completion_with_retry_retries_on_transport_error() -> None:
    """`httpx.TransportError` is in the retryable set."""
    llm = _make_llm(max_retries=1)
    mock_client = MagicMock()
    mock_client.create.side_effect = [
        httpx.ConnectError("refused"),
        _success_response(),
    ]
    llm.client = mock_client

    result = _completion_with_retry(llm, messages=[])

    assert result == _success_response()
    assert mock_client.create.call_count == 2
