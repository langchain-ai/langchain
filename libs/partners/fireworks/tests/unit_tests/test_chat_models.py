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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_fireworks import ChatFireworks
from langchain_fireworks.chat_models import (
    _acompletion_with_retry,
    _completion_with_retry,
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _format_message_content,
    _usage_to_metadata,
)

MODEL_NAME = "accounts/fireworks/models/test-model"


def _make_model(**kwargs: Any) -> ChatFireworks:
    defaults: dict[str, Any] = {"model": MODEL_NAME, "api_key": "fake-key"}
    defaults.update(kwargs)
    return ChatFireworks(**defaults)  # type: ignore[arg-type]


_STREAM_CHUNKS: list[dict[str, Any]] = [
    {
        "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}],
    },
    {
        "choices": [{"delta": {"content": "Hello"}, "index": 0}],
    },
    {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
    },
    # Final usage-only chunk (empty choices)
    {
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
]


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


def test_format_message_content_passthrough_string() -> None:
    """Plain string content is returned unchanged."""
    assert _format_message_content("hello") == "hello"


def test_format_message_content_translates_v1_image_block() -> None:
    """Canonical v1 image block is translated to OpenAI image_url + data URI."""
    blocks = [{"type": "image", "base64": "abc", "mime_type": "image/png"}]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_format_message_content_translates_v0_base64_image_block() -> None:
    """v0 source_type='base64' image block is translated."""
    blocks = [
        {
            "type": "image",
            "source_type": "base64",
            "data": "qqq",
            "mime_type": "image/png",
        }
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,qqq"}},
    ]


def test_format_message_content_passes_through_existing_image_url() -> None:
    """Already-OpenAI image_url blocks pass through unchanged."""
    blocks = [
        {"type": "image_url", "image_url": {"url": "https://example.com/y.png"}},
    ]

    formatted = _format_message_content(blocks)

    assert formatted == blocks


@pytest.mark.parametrize(
    "btype",
    [
        "tool_use",
        "thinking",
        "reasoning_content",
        "function_call",
        "code_interpreter_call",
    ],
)
def test_format_message_content_drops_unsupported_block_types(btype: str) -> None:
    """Block types not part of the OpenAI chat completions wire format are stripped."""
    blocks = [
        {"type": "text", "text": "visible"},
        {"type": btype, "foo": "bar"},
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [{"type": "text", "text": "visible"}]


def test_format_message_content_preserves_order_around_dropped_blocks() -> None:
    """Surviving blocks keep their order when interleaved drops are removed."""
    blocks = [
        {"type": "text", "text": "before"},
        {"type": "thinking", "thinking": "..."},
        {"type": "image", "base64": "abc", "mime_type": "image/png"},
        {"type": "tool_use", "name": "t", "input": {}},
        {"type": "text", "text": "after"},
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "after"},
    ]


def test_format_message_content_translates_v1_url_image_block() -> None:
    """v1 image block with a top-level URL maps to an OpenAI image_url block."""
    blocks = [{"type": "image", "url": "https://example.com/img.png"}]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]


def test_format_message_content_translates_v0_url_image_block() -> None:
    """v0 source_type=url image block is translated."""
    blocks = [
        {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/v0.png",
        }
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "image_url", "image_url": {"url": "https://example.com/v0.png"}},
    ]


def test_format_message_content_translates_anthropic_source_base64_image() -> None:
    """Legacy Anthropic-shape image with base64 source maps to a data URI."""
    blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc",
            },
        }
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_format_message_content_translates_anthropic_source_url_image() -> None:
    """Legacy Anthropic-shape image with url source maps to image_url."""
    blocks = [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/anthropic.png"},
        }
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/anthropic.png"},
        },
    ]


def test_format_message_content_translates_v1_audio_block() -> None:
    """v1 audio block is translated to OpenAI input_audio shape."""
    blocks = [{"type": "audio", "base64": "aGVsbG8=", "mime_type": "audio/wav"}]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
    ]


def test_format_message_content_translates_v1_file_block_base64() -> None:
    """v1 file block with base64 + filename maps to OpenAI file_data shape."""
    blocks = [
        {
            "type": "file",
            "base64": "JVBERi0=",
            "mime_type": "application/pdf",
            "filename": "x.pdf",
        }
    ]

    formatted = _format_message_content(blocks)

    assert formatted == [
        {
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,JVBERi0=",
                "filename": "x.pdf",
            },
        },
    ]


def test_convert_message_to_dict_translates_tool_message_image() -> None:
    """ToolMessage with a canonical image block lands as OpenAI image_url on the wire.

    Reproduces the failure mode where a tool that returns an image (e.g. a
    file-reader) hands back `content_blocks=[{"type": "image", ...}]` and the
    message round-trips into a Fireworks chat completions request.
    """
    tool_message = ToolMessage(
        content=[{"type": "image", "base64": "abc", "mime_type": "image/png"}],
        tool_call_id="call_1",
    )

    result = _convert_message_to_dict(tool_message)

    assert result == {
        "role": "tool",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
        "tool_call_id": "call_1",
    }


def test_convert_message_to_dict_translates_human_mixed_content() -> None:
    """HumanMessage with mixed text + image blocks translates correctly."""
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "what is this?"},
            {"type": "image", "base64": "xyz", "mime_type": "image/jpeg"},
        ]
    )

    result = _convert_message_to_dict(human_message)

    assert result == {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,xyz"},
            },
        ],
    }


def test_convert_message_to_dict_chat_message_uses_translator() -> None:
    """ChatMessage path also runs content through the formatter."""
    chat_message = ChatMessage(
        role="custom",
        content=[{"type": "image", "base64": "zz", "mime_type": "image/gif"}],
    )

    result = _convert_message_to_dict(chat_message)

    assert result == {
        "role": "custom",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/gif;base64,zz"}},
        ],
    }


def test_convert_message_to_dict_string_content_unchanged() -> None:
    """String content on common message types passes through unmodified."""
    assert _convert_message_to_dict(HumanMessage(content="hi"))["content"] == "hi"
    assert _convert_message_to_dict(SystemMessage(content="sys"))["content"] == "sys"
    assert (
        _convert_message_to_dict(ToolMessage(content="r1", tool_call_id="t"))["content"]
        == "r1"
    )


def test_convert_message_to_dict_translates_system_list_content() -> None:
    """SystemMessage with list content is routed through the formatter."""
    system_message = SystemMessage(
        content=[
            {"type": "text", "text": "rules"},
            {"type": "thinking", "thinking": "drop me"},
        ]
    )

    result = _convert_message_to_dict(system_message)

    assert result == {
        "role": "system",
        "content": [{"type": "text", "text": "rules"}],
    }


def test_convert_message_to_dict_translates_ai_message_image_content() -> None:
    """AIMessage with a canonical image block is translated, not forwarded raw."""
    ai_message = AIMessage(
        content=[
            {"type": "text", "text": "see attached"},
            {"type": "image", "base64": "abc", "mime_type": "image/png"},
        ]
    )

    result = _convert_message_to_dict(ai_message)

    assert result == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "see attached"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }


def test_convert_message_to_dict_propagates_translator_value_error() -> None:
    """Translator errors surface to callers instead of shipping bad payloads.

    Chat completions does not support file URLs; the translator raises rather
    than letting an unsupported block through.
    """
    bad_message = HumanMessage(
        content=[{"type": "file", "url": "https://example.com/doc.pdf"}]
    )

    with pytest.raises(ValueError, match="file URLs"):
        _convert_message_to_dict(bad_message)


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


def test_completion_with_retry_streaming_accepts_iterable_only_result() -> None:
    """Streaming setup accepts iterable-only custom client wrappers."""

    class _IterableOnlyStream:
        def __iter__(self) -> Any:
            yield {"id": 0, "choices": [{"delta": {"content": "one"}}]}
            yield {"id": 1, "choices": [{"delta": {"content": "two"}}]}

    llm = _make_llm(max_retries=0)
    mock_client = MagicMock()
    mock_client.create.return_value = _IterableOnlyStream()
    llm.client = mock_client

    chunks = list(_completion_with_retry(llm, messages=[], stream=True))

    assert [c["id"] for c in chunks] == [0, 1]
    assert mock_client.create.call_count == 1


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


async def test_acompletion_with_retry_streaming_accepts_async_iterable_only_result() -> (  # noqa: E501
    None
):
    """Async streaming setup accepts async-iterable-only custom wrappers."""

    class _AsyncIterableOnlyStream:
        def __aiter__(self) -> Any:
            async def _aiter() -> Any:
                yield {"id": 0, "choices": [{"delta": {"content": "one"}}]}
                yield {"id": 1, "choices": [{"delta": {"content": "two"}}]}

            return _aiter()

    llm = _make_llm(max_retries=0)
    mock_async = MagicMock()
    mock_async.acreate = MagicMock(return_value=_AsyncIterableOnlyStream())
    llm.async_client = mock_async

    agen = await _acompletion_with_retry(llm, messages=[], stream=True)
    chunks = [c async for c in agen]

    assert [c["id"] for c in chunks] == [0, 1]
    assert mock_async.acreate.call_count == 1


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


class TestUsageToMetadata:
    """Tests for the `_usage_to_metadata` helper."""

    def test_all_fields_present(self) -> None:
        result = _usage_to_metadata(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert result == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    def test_total_tokens_fallback_sums_input_and_output(self) -> None:
        """When provider omits total_tokens, sum input + output."""
        result = _usage_to_metadata({"prompt_tokens": 7, "completion_tokens": 3})
        assert result == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}

    def test_missing_fields_default_to_zero(self) -> None:
        result = _usage_to_metadata({})
        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class TestConvertChunkToMessageChunk:
    """Tests for `_convert_chunk_to_message_chunk` empty-choices handling."""

    def test_empty_choices_with_usage_returns_usage_chunk(self) -> None:
        chunk = {
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
        result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(result, AIMessageChunk)
        assert result.content == ""
        assert result.usage_metadata == {
            "input_tokens": 4,
            "output_tokens": 1,
            "total_tokens": 5,
        }

    def test_empty_choices_without_usage_logs_and_returns_blank(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunk: dict[str, Any] = {"choices": []}
        with caplog.at_level("DEBUG", logger="langchain_fireworks.chat_models"):
            result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(result, AIMessageChunk)
        assert result.content == ""
        assert result.usage_metadata is None
        assert any("no choices and no usage" in rec.message for rec in caplog.records)

    def test_missing_choices_key_treated_as_empty(self) -> None:
        """Provider may omit `choices` entirely on the final usage frame."""
        chunk = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(result, AIMessageChunk)
        assert result.usage_metadata == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }


class TestStreamUsage:
    """Tests for the `stream_usage` field and `stream_options` plumbing."""

    def test_stream_options_passed_by_default(self) -> None:
        model = _make_model()
        model.client = MagicMock()
        model.client.create.return_value = iter(list(_STREAM_CHUNKS))
        list(model.stream("Hello"))
        call_kwargs = model.client.create.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_stream_options_not_passed_when_disabled(self) -> None:
        model = _make_model(stream_usage=False)
        model.client = MagicMock()
        model.client.create.return_value = iter(list(_STREAM_CHUNKS))
        list(model.stream("Hello"))
        call_kwargs = model.client.create.call_args[1]
        assert "stream_options" not in call_kwargs

    def test_user_stream_options_in_model_kwargs_wins(self) -> None:
        """User-provided stream_options via model_kwargs overrides the default."""
        custom = {"include_usage": False}
        model = _make_model(model_kwargs={"stream_options": custom})
        model.client = MagicMock()
        model.client.create.return_value = iter(list(_STREAM_CHUNKS))
        list(model.stream("Hello"))
        call_kwargs = model.client.create.call_args[1]
        assert call_kwargs["stream_options"] == custom

    def test_usage_only_chunk_emits_usage_metadata(self) -> None:
        """The final empty-choices + usage chunk propagates as usage_metadata."""
        model = _make_model()
        model.client = MagicMock()
        model.client.create.return_value = iter(list(_STREAM_CHUNKS))
        chunks = list(model.stream("Hello"))
        usage_chunks = [c for c in chunks if c.usage_metadata]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage_metadata == {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        }

    async def test_astream_options_passed_by_default(self) -> None:
        model = _make_model()
        model.async_client = MagicMock()

        async def _aiter() -> Any:
            for c in _STREAM_CHUNKS:
                yield c

        model.async_client.acreate = MagicMock(return_value=_aiter())
        [chunk async for chunk in model.astream("Hello")]
        call_kwargs = model.async_client.acreate.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_astream_usage_only_chunk_emits_usage_metadata(self) -> None:
        model = _make_model()
        model.async_client = MagicMock()

        async def _aiter() -> Any:
            for c in _STREAM_CHUNKS:
                yield c

        model.async_client.acreate = MagicMock(return_value=_aiter())
        chunks = [chunk async for chunk in model.astream("Hello")]
        usage_chunks = [c for c in chunks if c.usage_metadata]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage_metadata == {
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        }
