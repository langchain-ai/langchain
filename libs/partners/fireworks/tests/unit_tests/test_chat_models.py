"""Unit tests for ChatFireworks."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_fireworks import ChatFireworks
from langchain_fireworks.chat_models import (
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
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
