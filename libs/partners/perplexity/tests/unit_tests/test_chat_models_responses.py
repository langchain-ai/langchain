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
    _convert_responses_usage,
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
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    assert llm._use_responses_api({"tools": [{"type": "web_search"}]}) is True


def test_instance_auto_detect_routes_to_chat_completions_for_plain_text() -> None:
    llm = ChatPerplexity(model="sonar", api_key="test")
    assert (
        llm._use_responses_api({"messages": [{"role": "user", "content": "hi"}]})
        is False
    )


def test_instance_explicit_true_overrides_auto_detect() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=True)
    assert llm._use_responses_api({"messages": []}) is True


def test_instance_explicit_false_overrides_auto_detect() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=False)
    assert llm._use_responses_api({"tools": [{"type": "web_search"}]}) is False


# ---------------------------------------------------------------------------
# Routing: full invoke path with mocked SDK clients
# ---------------------------------------------------------------------------


def _stub_responses_response(text: str = "ok") -> MagicMock:
    usage = _make_response_obj(input_tokens=11, output_tokens=22, total_tokens=33)
    return _make_response_obj(
        id="resp_123",
        model="sonar-pro",
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
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("hello")
    chat_create = llm.client.chat.completions.create

    result = llm.invoke("Find recent news", tools=[{"type": "web_search"}])

    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    llm.client.responses.create.assert_called_once()
    # Pin the original regression: the class-default `temperature=0.7` injected
    # via `_default_params` must NOT reach the Responses SDK call, either at
    # top level or inside `extra_body`.
    call_kwargs = llm.client.responses.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "temperature" not in call_kwargs.get("extra_body", {}) or {}
    chat_create.assert_not_called()


def test_invoke_routes_to_responses_when_previous_response_id_bound() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("continuation")
    chat_create = llm.client.chat.completions.create

    bound = llm.bind(previous_response_id="resp_abc")
    result = bound.invoke("continue please")

    assert isinstance(result, AIMessage)
    assert result.content == "continuation"
    llm.client.responses.create.assert_called_once()
    call_kwargs = llm.client.responses.create.call_args.kwargs
    # `previous_response_id` is forwarded via `extra_body` because the typed
    # Perplexity SDK signature does not yet expose it.
    assert call_kwargs["extra_body"]["previous_response_id"] == "resp_abc"
    assert "previous_response_id" not in call_kwargs
    # Class-default temperature must not leak through to the Responses call.
    assert "temperature" not in call_kwargs
    assert "temperature" not in (call_kwargs.get("extra_body") or {})
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
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=True)
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("forced")

    result = llm.invoke("plain prompt")

    assert isinstance(result, AIMessage)
    assert result.content == "forced"
    llm.client.responses.create.assert_called_once()
    # Class-default temperature must not leak through to the Responses call —
    # this is the exact scenario that produced the original `TypeError`.
    call_kwargs = llm.client.responses.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "temperature" not in (call_kwargs.get("extra_body") or {})
    llm.client.chat.completions.create.assert_not_called()


def test_invoke_drops_explicit_stop_on_responses_branch_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`stop=` from the standard `BaseChatModel.invoke` path must be dropped."""
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=True)
    llm.client = MagicMock()
    llm.client.responses.create.return_value = _stub_responses_response("ok")

    with caplog.at_level("WARNING", logger="langchain_perplexity.chat_models"):
        llm.invoke("hi", stop=["END"])

    call_kwargs = llm.client.responses.create.call_args.kwargs
    assert "stop" not in call_kwargs
    assert "stop_sequences" not in call_kwargs
    assert "stop" not in (call_kwargs.get("extra_body") or {})
    # Functional drop emits a discoverable warning so users see the no-op.
    assert any("stop" in record.message for record in caplog.records)


def test_invoke_use_responses_api_false_forces_chat_completions_branch() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=False)
    llm.client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="from chat completions", tool_calls=None))
    ]
    mock_response.model = "sonar-pro"
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
        model="sonar-pro",
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
    assert message.response_metadata["model"] == "sonar-pro"
    assert message.response_metadata["status"] == "completed"
    assert message.additional_kwargs["citations"] == ["https://example.com"]
    assert "citations" not in message.response_metadata
    assert "images" not in message.additional_kwargs


def test_convert_responses_to_chat_result_function_call_to_tool_calls() -> None:
    function_call_item = _make_response_obj(
        type="function_call",
        name="get_weather",
        arguments=json.dumps({"city": "Paris"}),
        call_id="call_42",
    )
    response = _make_response_obj(
        id="resp_abc",
        model="sonar-pro",
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
        model="sonar-pro",
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
        model="sonar-pro",
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


def test_stream_event_conversion_completed_surfaces_perplexity_extras() -> None:
    response = _make_response_obj(
        id="resp_extras_stream",
        model="sonar-pro",
        status="completed",
        object="response",
        usage=None,
        citations=["https://example.com"],
        images=[{"url": "https://example.com/img.png"}],
        related_questions=["What about X?"],
        search_results=[{"title": "T"}],
        videos=[{"url": "https://example.com/v.mp4"}],
        reasoning_steps=[{"step": "thinking"}],
    )
    event = _make_event("response.completed", response=response)
    chunk = _convert_responses_stream_event_to_chunk(event)
    assert chunk is not None
    assert isinstance(chunk.message, AIMessageChunk)
    for key in (
        "citations",
        "images",
        "related_questions",
        "search_results",
        "videos",
        "reasoning_steps",
    ):
        assert key in chunk.message.additional_kwargs
        assert key not in chunk.message.response_metadata


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
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=True)
    llm.client = MagicMock()

    usage = _make_response_obj(input_tokens=2, output_tokens=6, total_tokens=8)
    completed_response = _make_response_obj(
        id="resp_stream",
        model="sonar-pro",
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

    # Class-default temperature must not leak into the streaming call.
    call_kwargs = llm.client.responses.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "temperature" not in (call_kwargs.get("extra_body") or {})
    text_chunks = [c for c in chunks if c.content]
    assert "".join(c.content for c in text_chunks) == "Hello world"  # type: ignore[misc]
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
    llm = ChatPerplexity(model="sonar-pro", api_key="test", use_responses_api=True)

    usage = _make_response_obj(input_tokens=3, output_tokens=9, total_tokens=12)
    completed_response = _make_response_obj(
        id="resp_async",
        model="sonar-pro",
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

    # Class-default temperature must not leak into the async streaming call.
    call_kwargs = llm.async_client.responses.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "temperature" not in (call_kwargs.get("extra_body") or {})

    text = "".join(c.content for c in collected if c.content)  # type: ignore[misc]
    assert text == "foobar"
    usage_chunks = [c for c in collected if c.usage_metadata]
    assert usage_chunks
    final_usage = usage_chunks[-1].usage_metadata
    assert final_usage is not None
    assert final_usage["input_tokens"] == 3
    assert final_usage["output_tokens"] == 9


# ---------------------------------------------------------------------------
# Auto-detection: input/include/instructions/previous_response_id + mixed tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["input", "include", "instructions", "previous_response_id"],
)
def test_module_use_responses_api_detects_each_responses_only_field(key: str) -> None:
    assert _use_responses_api({key: "value"}) is True


def test_module_use_responses_api_detects_mixed_function_and_builtin_tools() -> None:
    assert (
        _use_responses_api(
            {
                "tools": [
                    {"type": "function", "function": {"name": "foo"}},
                    {"type": "web_search"},
                ]
            }
        )
        is True
    )


def test_module_use_responses_api_empty_tools_list_is_false() -> None:
    assert _use_responses_api({"tools": []}) is False


# ---------------------------------------------------------------------------
# _to_responses_payload translation
# ---------------------------------------------------------------------------


def test_to_responses_payload_renames_and_drops_keys() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    payload = llm._to_responses_payload(
        [{"role": "user", "content": "hi"}],
        {
            "model": "sonar-pro",
            "max_tokens": 128,
            "temperature": 0.4,  # Chat-Completions-only → dropped.
            "stream": True,
            "top_p": None,  # None values are dropped.
            "top_k": 5,  # Chat-Completions-only → dropped.
            "tool_choice": "auto",  # Chat-Completions-only → dropped.
            "metadata": {"trace": "x"},  # Chat-Completions-only → dropped.
            "search_mode": "academic",  # Perplexity-specific → extra_body.
            "return_images": True,
        },
    )

    assert payload["input"] == [{"role": "user", "content": "hi"}]
    assert payload["model"] == "sonar-pro"
    assert payload["max_output_tokens"] == 128
    assert "max_tokens" not in payload
    assert payload["stream"] is True
    for dropped in ("temperature", "top_p", "top_k", "tool_choice", "metadata"):
        assert dropped not in payload
    assert "messages" not in payload
    extra_body = payload["extra_body"]
    for dropped in ("temperature", "top_p", "top_k", "tool_choice", "metadata"):
        assert dropped not in extra_body
    assert extra_body == {
        "search_mode": "academic",
        "return_images": True,
    }


def test_to_responses_payload_drops_stop() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    payload = llm._to_responses_payload(
        [{"role": "user", "content": "hi"}],
        {"model": "sonar-pro", "stop": ["END"]},
    )
    # Perplexity Responses API does not support stop sequences; dropped at the
    # boundary rather than forwarded as `stop_sequences`.
    assert "stop" not in payload
    assert "stop_sequences" not in payload
    assert "extra_body" not in payload


def test_to_responses_payload_drops_model_when_preset_set() -> None:
    """`model` must be dropped when a `preset` is supplied.

    Perplexity's Agent API validates `model` strictly and rejects bare
    Chat-Completions names like `sonar-pro` even when a preset is also set.
    """
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    payload = llm._to_responses_payload(
        [{"role": "user", "content": "hi"}],
        {"model": "sonar-pro", "preset": "sonar-pro"},
    )
    assert payload["preset"] == "sonar-pro"
    assert "model" not in payload


def test_to_responses_payload_silently_drops_class_default_temperature() -> None:
    """The class default `temperature=0.7` must not warn — it's injected on
    every call regardless of user intent, so warning would spam.
    """
    import logging

    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    assert "temperature" not in llm.model_fields_set
    logger = logging.getLogger("langchain_perplexity.chat_models")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        payload = llm._to_responses_payload(
            [{"role": "user", "content": "hi"}],
            {"model": "sonar-pro", "temperature": 0.7},
        )
    finally:
        logger.removeHandler(handler)
    assert "temperature" not in payload
    assert "temperature" not in payload.get("extra_body", {})
    assert not [r for r in records if r.levelno >= logging.WARNING]


def test_to_responses_payload_warns_when_user_set_temperature_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicitly-set temperature must warn so the no-op is discoverable."""
    llm = ChatPerplexity(model="sonar-pro", api_key="test", temperature=0.2)
    assert "temperature" in llm.model_fields_set
    with caplog.at_level("WARNING", logger="langchain_perplexity.chat_models"):
        payload = llm._to_responses_payload(
            [{"role": "user", "content": "hi"}],
            {"model": "sonar-pro", "temperature": 0.2},
        )
    assert "temperature" not in payload
    assert any("temperature" in record.message for record in caplog.records)


def test_to_responses_payload_warns_on_functional_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`tool_choice`, `stop`, `metadata` are functional; their silent drop
    would be a footgun, so we surface a warning.
    """
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    with caplog.at_level("WARNING", logger="langchain_perplexity.chat_models"):
        llm._to_responses_payload(
            [{"role": "user", "content": "hi"}],
            {
                "model": "sonar-pro",
                "tool_choice": "auto",
                "stop": ["END"],
                "metadata": {"trace_id": "x"},
            },
        )
    assert any(
        all(k in record.message for k in ("tool_choice", "stop", "metadata"))
        for record in caplog.records
    )


def test_to_responses_payload_routes_previous_response_id_via_extra_body() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    payload = llm._to_responses_payload(
        [{"role": "user", "content": "continue"}],
        {
            "model": "sonar-pro",
            "previous_response_id": "resp_abc",
            "include": ["citations"],
        },
    )
    assert payload["extra_body"] == {
        "previous_response_id": "resp_abc",
        "include": ["citations"],
    }
    assert "previous_response_id" not in {k for k in payload if k != "extra_body"}


def test_to_responses_payload_raises_for_non_dict_extra_body() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    with pytest.raises(TypeError, match="extra_body"):
        llm._to_responses_payload(
            [{"role": "user", "content": "hi"}],
            {
                "model": "sonar-pro",
                "extra_body": "not-a-dict",
                "search_mode": "academic",
            },
        )


def test_to_responses_payload_preserves_existing_extra_body() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    payload = llm._to_responses_payload(
        [{"role": "user", "content": "hi"}],
        {
            "model": "sonar-pro",
            "extra_body": {"caller_set": True},
            "search_mode": "academic",
        },
    )
    assert payload["extra_body"] == {"caller_set": True, "search_mode": "academic"}


# ---------------------------------------------------------------------------
# Usage conversion edge cases
# ---------------------------------------------------------------------------


def test_convert_responses_usage_returns_none_when_usage_missing() -> None:
    assert _convert_responses_usage(None) is None


def test_convert_responses_usage_returns_none_when_tokens_missing() -> None:
    usage = _make_response_obj(input_tokens=None, output_tokens=None, total_tokens=None)
    assert _convert_responses_usage(usage) is None


def test_convert_responses_usage_derives_total_when_absent() -> None:
    usage = _make_response_obj(input_tokens=5, output_tokens=7, total_tokens=None)
    result = _convert_responses_usage(usage)
    assert result is not None
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 12


# ---------------------------------------------------------------------------
# Error and edge cases in conversion / streaming
# ---------------------------------------------------------------------------


def test_convert_responses_to_chat_result_malformed_json_arguments() -> None:
    function_call_item = _make_response_obj(
        type="function_call",
        name="get_weather",
        arguments="{not valid json",
        call_id="call_99",
    )
    response = _make_response_obj(
        id="resp_bad_json",
        model="sonar-pro",
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
    assert message.tool_calls[0]["args"] == {"__raw_arguments__": "{not valid json"}


def test_responses_extras_land_on_additional_kwargs() -> None:
    response = _make_response_obj(
        id="resp_extras",
        model="sonar-pro",
        status="completed",
        object="response",
        output_text="hi",
        output=[],
        usage=None,
        citations=["https://example.com"],
        images=[{"url": "https://example.com/img.png"}],
        related_questions=["What about X?"],
        search_results=[{"title": "T"}],
        videos=[{"url": "https://example.com/v.mp4"}],
        reasoning_steps=[{"step": "thinking"}],
    )
    message = _convert_responses_to_chat_result(response).generations[0].message
    assert isinstance(message, AIMessage)
    for key in (
        "citations",
        "images",
        "related_questions",
        "search_results",
        "videos",
        "reasoning_steps",
    ):
        assert key in message.additional_kwargs
        assert key not in message.response_metadata


def test_stream_event_conversion_error_surfaces_structured_fields() -> None:
    error = _make_response_obj(
        message="rate limited",
        code="rate_limit_exceeded",
        type="rate_limit",
        param=None,
    )
    event = _make_event("response.error", error=error, request_id="req_abc")
    with pytest.raises(RuntimeError) as exc_info:
        _convert_responses_stream_event_to_chunk(event)
    message = str(exc_info.value)
    assert "rate limited" in message
    assert "code=rate_limit_exceeded" in message
    assert "type=rate_limit" in message
    assert "request_id=req_abc" in message


def test_stream_event_conversion_error_uses_default_message_when_missing() -> None:
    event = MagicMock(spec_set=["type"])
    event.type = "response.error"
    with pytest.raises(RuntimeError, match="Perplexity Responses API stream error"):
        _convert_responses_stream_event_to_chunk(event)


# ---------------------------------------------------------------------------
# Async non-streaming Responses path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ainvoke_routes_to_responses_when_builtin_tool_in_payload() -> None:
    llm = ChatPerplexity(model="sonar-pro", api_key="test")
    llm.async_client = MagicMock()
    llm.async_client.responses.create = AsyncMock(
        return_value=_stub_responses_response("async-ok")
    )
    chat_create = llm.async_client.chat.completions.create

    result = await llm.ainvoke("Find recent news", tools=[{"type": "web_search"}])

    assert isinstance(result, AIMessage)
    assert result.content == "async-ok"
    llm.async_client.responses.create.assert_awaited_once()
    # Class-default temperature must not leak through the async invoke path.
    call_kwargs = llm.async_client.responses.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "temperature" not in (call_kwargs.get("extra_body") or {})
    chat_create.assert_not_called()
