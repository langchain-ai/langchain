"""Unit tests for the openrouter native stream-events converter."""

import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_openrouter import ChatOpenRouter
from langchain_openrouter._stream_events import (
    aconvert_openrouter_stream,
    convert_openrouter_stream,
)

if "OPENROUTER_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = "fake-key"


def _reasoning_text_tool() -> list[dict]:
    m = "anthropic/claude-3.5-sonnet"
    return [
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning": "Let me "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "think."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hi "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "there"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "t1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ' "Paris"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "model": m,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    ]


def test_convert_openrouter_stream_lifecycle() -> None:
    """Reasoning, text, and tool-call deltas surface as finished blocks."""
    events: list[Any] = list(convert_openrouter_stream(iter(_reasoning_text_tool())))
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "openrouter"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    assert (
        cast("dict[str, Any]", finishes[0]["content"])["reasoning"] == "Let me think."
    )
    assert cast("dict[str, Any]", finishes[1]["content"])["text"] == "Hi there"
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    assert events[-1]["event"] == "message-finish"
    assert events[-1]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_convert_openrouter_stream_error_chunk_raises() -> None:
    """A no-choices chunk carrying an `error` payload raises `ValueError`."""
    chunks = [{"choices": [], "error": {"message": "rate limited", "code": 429}}]
    with pytest.raises(ValueError, match="OpenRouter API returned an error"):
        list(convert_openrouter_stream(iter(chunks)))


def test_convert_openrouter_stream_surfaces_cost() -> None:
    """Cost/native-finish/model metadata reach `message-finish`, like the bridge.

    The final chunk carries both `finish_reason` metadata and usage/cost data;
    both must land in the `message-finish` response metadata so switching to the
    v3 path does not silently drop OpenRouter's documented cost surfacing.
    """
    cost_details = {
        "upstream_inference_cost": 7.745e-05,
        "upstream_inference_prompt_cost": 8.95e-06,
        "upstream_inference_completions_cost": 6.85e-05,
    }
    chunks: list[dict] = [
        {
            "model": "openai/gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
        },
        {
            "model": "openai/gpt-4o-mini",
            "id": "gen-cost-stream",
            "system_fingerprint": "fp_abc",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 7.5e-05,
                "cost_details": cost_details,
            },
        },
    ]
    events: list[Any] = list(convert_openrouter_stream(iter(chunks)))
    assert_valid_event_stream(events)

    finish = events[-1]
    assert finish["event"] == "message-finish"
    meta = finish["metadata"]
    assert meta["cost"] == 7.5e-05
    assert meta["cost_details"] == cost_details
    assert meta["finish_reason"] == "stop"
    assert meta["native_finish_reason"] == "end_turn"
    assert meta["model_name"] == "openai/gpt-4o-mini"
    assert meta["system_fingerprint"] == "fp_abc"
    assert meta["id"] == "gen-cost-stream"
    assert finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def _parallel_tool_calls() -> list[dict]:
    """Two tool calls interleaved at index 0 and 1 across chunks."""
    m = "anthropic/claude-3.5-sonnet"
    return [
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":',
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"zone":',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ' "Paris"}'}},
                            {"index": 1, "function": {"arguments": ' "UTC"}'}},
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]


def test_convert_openrouter_stream_parallel_tool_calls() -> None:
    """Args at distinct `tool:{idx}` keys finalize as separate tool calls."""
    events: list[Any] = list(convert_openrouter_stream(iter(_parallel_tool_calls())))
    assert_valid_event_stream(events)

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["tool_call", "tool_call"]

    first = cast("dict[str, Any]", finishes[0]["content"])
    second = cast("dict[str, Any]", finishes[1]["content"])
    assert first["name"] == "get_weather"
    assert first["args"] == {"city": "Paris"}
    assert second["name"] == "get_time"
    assert second["args"] == {"zone": "UTC"}


async def test_aconvert_openrouter_stream_lifecycle() -> None:
    """Async twin of `test_convert_openrouter_stream_lifecycle`."""

    async def _araw() -> Any:
        for chunk in _reasoning_text_tool():
            yield chunk

    events: list[Any] = [e async for e in aconvert_openrouter_stream(_araw())]
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "openrouter"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    assert (
        cast("dict[str, Any]", finishes[0]["content"])["reasoning"] == "Let me think."
    )
    assert cast("dict[str, Any]", finishes[1]["content"])["text"] == "Hi there"
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    assert events[-1]["event"] == "message-finish"
    assert events[-1]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


async def test_aconvert_openrouter_stream_error_chunk_raises() -> None:
    """Async twin of `test_convert_openrouter_stream_error_chunk_raises`."""

    async def _araw() -> Any:
        yield {"choices": [], "error": {"message": "rate limited", "code": 429}}

    with pytest.raises(ValueError, match="OpenRouter API returned an error"):
        [e async for e in aconvert_openrouter_stream(_araw())]


def test_openrouter_stream_events_v3_lifecycle() -> None:
    """Drive `stream_events(version="v3")` through the native sync hook.

    Confirms the model-level path threads the LangChain run id onto
    `message-start` (non-empty, unlike the converter's empty default) and
    surfaces reasoning/text/tool blocks with `model_provider == "openrouter"`.
    """
    llm = ChatOpenRouter(model="foo")
    mock_client = MagicMock()

    def mock_send(*_a: Any, **_k: Any) -> Any:
        for chunk in _reasoning_text_tool():
            mock = MagicMock()
            mock.model_dump.return_value = chunk
            yield mock

    mock_client.chat.send = mock_send

    with patch.object(llm, "client", mock_client):
        events = list(llm.stream_events("Test query", version="v3"))

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]  # core seeds a non-empty LangChain run id
    assert message_start["metadata"]["provider"] == "openrouter"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "openrouter"


async def test_openrouter_astream_events_v3_lifecycle() -> None:
    """Async twin of `test_openrouter_stream_events_v3_lifecycle`."""
    llm = ChatOpenRouter(model="foo")

    async def _araw() -> Any:
        for chunk in _reasoning_text_tool():
            mock = MagicMock()
            mock.model_dump.return_value = chunk
            yield mock

    mock_client = MagicMock()
    mock_client.chat.send_async = AsyncMock(return_value=_araw())

    with patch.object(llm, "client", mock_client):
        stream = await llm.astream_events("Test query", version="v3")
        events = [e async for e in stream]

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]
    assert message_start["metadata"]["provider"] == "openrouter"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["metadata"]["model_provider"] == "openrouter"
