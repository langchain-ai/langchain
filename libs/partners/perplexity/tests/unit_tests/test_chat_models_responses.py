"""Tests for the Responses (Agent) API integration in `ChatPerplexity`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_perplexity import ChatPerplexity
from langchain_perplexity.chat_models import (
    _convert_responses_stream_event_to_chunk,
    _convert_responses_to_chat_result,
    _use_responses_api,
)


def _make_response_obj(**attrs: Any) -> MagicMock:
    """Create a MagicMock that mimics the Perplexity Responses SDK object."""
    obj = MagicMock(spec_set=list(attrs.keys()))
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _make_event(event_type: str, **attrs: Any) -> MagicMock:
    obj = MagicMock(spec_set=["type", *attrs.keys()])
    obj.type = event_type
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# ---------------------------------------------------------------------------
# Module-level _use_responses_api helper
# ---------------------------------------------------------------------------


def test_module_use_responses_api_detects_builtin_tool() -> None:
    assert _use_responses_api({"tools": [{"type": "web_search"}]}) is True


def test_module_use_responses_api_ignores_function_tool() -> None:
    assert (
        _use_responses_api(
            {"tools": [{"type": "function", "function": {"name": "foo"}}]}
        )
        is False
    )


def test_module_use_responses_api_detects_previous_response_id() -> None:
    assert _use_responses_api({"previous_response_id": "resp_abc"}) is True


def test_module_use_responses_api_detects_instructions() -> None:
    assert _use_responses_api({"instructions": "Be brief"}) is True


def test_module_use_responses_api_returns_false_for_plain_payload() -> None:
    assert _use_responses_api({"temperature": 0.7}) is False


# ---------------------------------------------------------------------------
# Instance _use_responses_api method (auto-detect + explicit override)
# ---------------------------------------------------------------------------


def test_instance_auto_detect_routes_to_responses_for_builtin_tool() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test")
    assert llm._use_responses_api({"tools": [{"type": "web_search"}]}) is True


def test_instance_auto_detect_routes_to_chat_completions_for_plain_text() -> None:
    llm = ChatPerplexity(model="sonar", api_key="test")
    assert (
        llm._use_responses_api({"messages": [{"role": "user", "content": "hi"}]})
        is False
    )


def test_instance_explicit_true_overrides_auto_detect() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test", use_responses_api=True)
    assert llm._use_responses_api({"messages": []}) is True


def test_instance_explicit_false_overrides_auto_detect() -> None:
    llm = ChatPerplexity(
        model="openai/gpt-5.4", api_key="test", use_responses_api=False
    )
    assert llm._use_responses_api({"tools": [{"type": "web_search"}]}) is False


# ---------------------------------------------------------------------------
# Routing: full invoke path with mocked SDK clients
# ---------------------------------------------------------------------------


def _stub_responses_response(text: str = "ok") -> MagicMock:
    usage = _make_response_obj(input_tokens=11, output_tokens=22, total_tokens=33)
    return _make_response_obj(
        id="resp_123",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        output_text=text,
        output=[],
        usage=usage,
        citations=None,
        images=None,
        related_questions=None,
        search_results=None,
    )


def test_invoke_routes_to_responses_when_builtin_tool_in_payload() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test")
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("hello")
    chat_create = llm.client.chat.completions.create

    result = llm.invoke("Find recent news", tools=[{"type": "web_search"}])

    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    llm.client.responses.create.assert_called_once()
    chat_create.assert_not_called()


def test_invoke_routes_to_responses_when_previous_response_id_bound() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test")
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("continuation")
    chat_create = llm.client.chat.completions.create

    bound = llm.bind(previous_response_id="resp_abc")
    result = bound.invoke("continue please")

    assert isinstance(result, AIMessage)
    assert result.content == "continuation"
    llm.client.responses.create.assert_called_once()
    call_kwargs = llm.client.responses.create.call_args.kwargs
    assert call_kwargs["previous_response_id"] == "resp_abc"
    chat_create.assert_not_called()


def test_invoke_routes_to_chat_completions_for_plain_text() -> None:
    llm = ChatPerplexity(model="sonar", api_key="test")
    llm.client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="plain response", tool_calls=None))
    ]
    mock_response.model = "sonar"
    mock_response.usage = None
    for attr in (
        "videos",
        "reasoning_steps",
        "citations",
        "search_results",
        "images",
        "related_questions",
    ):
        setattr(mock_response, attr, None)
    llm.client.chat.completions.create.return_value = mock_response

    result = llm.invoke("Hello")

    assert isinstance(result, AIMessage)
    assert result.content == "plain response"
    llm.client.chat.completions.create.assert_called_once()
    llm.client.responses.create.assert_not_called()


def test_invoke_use_responses_api_true_forces_responses_branch() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test", use_responses_api=True)
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("forced")

    result = llm.invoke("plain prompt")

    assert isinstance(result, AIMessage)
    assert result.content == "forced"
    llm.client.responses.create.assert_called_once()
    llm.client.chat.completions.create.assert_not_called()


def test_invoke_use_responses_api_false_forces_chat_completions_branch() -> None:
    llm = ChatPerplexity(
        model="openai/gpt-5.4", api_key="test", use_responses_api=False
    )
    llm.client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="from chat completions", tool_calls=None))
    ]
    mock_response.model = "openai/gpt-5.4"
    mock_response.usage = None
    for attr in (
        "videos",
        "reasoning_steps",
        "citations",
        "search_results",
        "images",
        "related_questions",
    ):
        setattr(mock_response, attr, None)
    llm.client.chat.completions.create.return_value = mock_response

    result = llm.invoke("hi", tools=[{"type": "web_search"}])

    assert isinstance(result, AIMessage)
    assert result.content == "from chat completions"
    llm.client.chat.completions.create.assert_called_once()
    llm.client.responses.create.assert_not_called()


# ---------------------------------------------------------------------------
# Response conversion: text + annotations + usage_metadata + response_metadata
# ---------------------------------------------------------------------------


def test_convert_responses_to_chat_result_basic_fields() -> None:
    annotation = {
        "type": "url_citation",
        "url": "https://example.com",
        "title": "Example",
        "start_index": 0,
        "end_index": 5,
    }
    text_block = _make_response_obj(
        type="output_text", text="Hello world", annotations=[annotation]
    )
    message_item = _make_response_obj(
        type="message", role="assistant", content=[text_block]
    )
    usage = _make_response_obj(input_tokens=12, output_tokens=34, total_tokens=46)
    response = _make_response_obj(
        id="resp_xyz",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        output_text="Hello world",
        output=[message_item],
        usage=usage,
        citations=["https://example.com"],
        images=None,
        related_questions=None,
        search_results=None,
    )

    result = _convert_responses_to_chat_result(response)
    message = result.generations[0].message

    assert isinstance(message, AIMessage)
    assert message.content == "Hello world"
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 12
    assert message.usage_metadata["output_tokens"] == 34
    assert message.usage_metadata["total_tokens"] == 46
    assert message.response_metadata["id"] == "resp_xyz"
    assert message.response_metadata["model"] == "openai/gpt-5.4"
    assert message.response_metadata["status"] == "completed"
    assert message.response_metadata["citations"] == ["https://example.com"]


def test_convert_responses_to_chat_result_function_call_to_tool_calls() -> None:
    function_call_item = _make_response_obj(
        type="function_call",
        name="get_weather",
        arguments=json.dumps({"city": "Paris"}),
        call_id="call_42",
    )
    response = _make_response_obj(
        id="resp_abc",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        output_text="",
        output=[function_call_item],
        usage=None,
        citations=None,
        images=None,
        related_questions=None,
        search_results=None,
    )

    result = _convert_responses_to_chat_result(response)
    message = result.generations[0].message

    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1
    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call["args"] == {"city": "Paris"}
    assert tool_call["id"] == "call_42"


def test_convert_responses_to_chat_result_falls_back_to_output_content() -> None:
    text_block = _make_response_obj(type="output_text", text="fallback")
    message_item = _make_response_obj(
        type="message", role="assistant", content=[text_block]
    )
    response = _make_response_obj(
        id="resp_fb",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        output_text="",
        output=[message_item],
        usage=None,
        citations=None,
        images=None,
        related_questions=None,
        search_results=None,
    )

    result = _convert_responses_to_chat_result(response)
    assert result.generations[0].message.content == "fallback"


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------


def test_stream_event_conversion_for_text_delta() -> None:
    event = _make_event("response.output_text.delta", delta="Hello")
    chunk = _convert_responses_stream_event_to_chunk(event)
    assert chunk is not None
    assert isinstance(chunk.message, AIMessageChunk)
    assert chunk.message.content == "Hello"


def test_stream_event_conversion_for_completed_includes_usage() -> None:
    usage = _make_response_obj(input_tokens=4, output_tokens=8, total_tokens=12)
    response = _make_response_obj(
        id="resp_done",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        usage=usage,
    )
    event = _make_event("response.completed", response=response)
    chunk = _convert_responses_stream_event_to_chunk(event)
    assert chunk is not None
    assert isinstance(chunk.message, AIMessageChunk)
    assert chunk.message.usage_metadata is not None
    assert chunk.message.usage_metadata["input_tokens"] == 4
    assert chunk.message.usage_metadata["output_tokens"] == 8
    assert chunk.message.usage_metadata["total_tokens"] == 12


def test_stream_event_conversion_returns_none_for_unknown_event() -> None:
    event = _make_event("response.output_text.done")
    assert _convert_responses_stream_event_to_chunk(event) is None


def test_stream_event_conversion_raises_on_error_event() -> None:
    error = _make_response_obj(message="boom")
    event = _make_event("response.error", error=error)
    with pytest.raises(RuntimeError, match="boom"):
        _convert_responses_stream_event_to_chunk(event)


# ---------------------------------------------------------------------------
# Streaming end-to-end through the sync stream() entry point
# ---------------------------------------------------------------------------


def test_stream_yields_text_chunks_and_final_usage() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test", use_responses_api=True)
    llm.client = MagicMock()

    usage = _make_response_obj(input_tokens=2, output_tokens=6, total_tokens=8)
    completed_response = _make_response_obj(
        id="resp_stream",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        usage=usage,
    )
    events = [
        _make_event("response.output_text.delta", delta="Hello "),
        _make_event("response.output_text.delta", delta="world"),
        _make_event("response.completed", response=completed_response),
    ]
    llm.client.responses.create.return_value = iter(events)

    chunks = list(llm.stream("greet me"))

    text_chunks = [c for c in chunks if c.content]
    assert "".join(c.content for c in text_chunks) == "Hello world"
    usage_chunks = [
        c for c in chunks if isinstance(c, AIMessageChunk) and c.usage_metadata
    ]
    assert usage_chunks, "expected at least one chunk with usage_metadata"
    final_usage = usage_chunks[-1].usage_metadata
    assert final_usage is not None
    assert final_usage["input_tokens"] == 2
    assert final_usage["output_tokens"] == 6


@pytest.mark.asyncio
async def test_astream_yields_text_chunks_and_final_usage() -> None:
    llm = ChatPerplexity(model="openai/gpt-5.4", api_key="test", use_responses_api=True)

    usage = _make_response_obj(input_tokens=3, output_tokens=9, total_tokens=12)
    completed_response = _make_response_obj(
        id="resp_async",
        model="openai/gpt-5.4",
        status="completed",
        object="response",
        usage=usage,
    )
    events = [
        _make_event("response.output_text.delta", delta="foo"),
        _make_event("response.output_text.delta", delta="bar"),
        _make_event("response.completed", response=completed_response),
    ]

    class _AsyncIter:
        def __init__(self, items: list[Any]) -> None:
            self._items = list(items)

        def __aiter__(self) -> _AsyncIter:
            return self

        async def __anext__(self) -> Any:
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    llm.async_client = MagicMock()
    llm.async_client.responses.create = AsyncMock(return_value=_AsyncIter(events))

    collected: list[AIMessageChunk] = []
    async for chunk in llm.astream("greet me"):
        assert isinstance(chunk, AIMessageChunk)
        collected.append(chunk)

    text = "".join(c.content for c in collected if c.content)
    assert text == "foobar"
    usage_chunks = [c for c in collected if c.usage_metadata]
    assert usage_chunks
    final_usage = usage_chunks[-1].usage_metadata
    assert final_usage is not None
    assert final_usage["input_tokens"] == 3
    assert final_usage["output_tokens"] == 9
