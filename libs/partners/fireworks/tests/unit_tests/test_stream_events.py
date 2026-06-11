"""Unit tests for the Fireworks native stream-events converter."""

import os
from typing import Any, cast
from unittest.mock import MagicMock, patch

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_fireworks import ChatFireworks
from langchain_fireworks._stream_events import (
    aconvert_fireworks_stream,
    convert_fireworks_stream,
)

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = "fake-key"


def _reasoning_text_tool() -> list[dict]:
    m = "accounts/fireworks/models/deepseek-r1"
    return [
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "reasoning_content": "Let me "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "think."},
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
        # Final no-choices chunk carrying usage (Fireworks, not under x_groq)
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


def test_convert_fireworks_stream_lifecycle() -> None:
    events: list[Any] = list(convert_fireworks_stream(iter(_reasoning_text_tool())))
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "fireworks"
    assert events[0]["metadata"]["model"] == "accounts/fireworks/models/deepseek-r1"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["reasoning", "text", "tool_call"]
    assert (
        cast("dict[str, Any]", finishes[0]["content"])["reasoning"] == "Let me think."
    )
    assert cast("dict[str, Any]", finishes[1]["content"])["text"] == "Hi there"
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_fireworks_stream_events_v3_lifecycle() -> None:
    """Drive `stream_events(version="v3")` through the native sync hook.

    Confirms the model-level path threads the LangChain run id onto
    `message-start` (non-empty, unlike the converter's empty default) and
    surfaces reasoning/text/tool blocks with `model_provider == "fireworks"`.
    """
    llm = ChatFireworks(model="accounts/fireworks/models/deepseek-r1")
    mock_client = MagicMock()

    def mock_create(*_a: Any, **_k: Any) -> Any:
        return iter(_reasoning_text_tool())

    mock_client.create = mock_create

    with patch.object(llm, "client", mock_client):
        events = list(llm.stream_events("Test query", version="v3"))

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]  # core seeds a non-empty LangChain run id
    assert message_start["metadata"]["provider"] == "fireworks"

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
    assert message_finish["metadata"]["model_provider"] == "fireworks"


async def test_fireworks_astream_events_v3_lifecycle() -> None:
    """Async twin of `test_fireworks_stream_events_v3_lifecycle`."""
    llm = ChatFireworks(model="accounts/fireworks/models/deepseek-r1")

    async def _araw() -> Any:
        for chunk in _reasoning_text_tool():
            yield chunk

    # `_acompletion_with_retry` does `await llm.async_client.create(**kwargs)`
    # (1.x SDK: `create()` is a coroutine resolving to an `AsyncStream`), so
    # the mock must be a coroutine returning an async iterator.
    async def mock_create(*a: Any, **k: Any) -> Any:
        return _araw()

    mock_client = MagicMock()
    mock_client.create = mock_create

    with patch.object(llm, "async_client", mock_client):
        stream = await llm.astream_events("Test query", version="v3")
        events = [e async for e in stream]

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]
    assert message_start["metadata"]["provider"] == "fireworks"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["metadata"]["model_provider"] == "fireworks"


async def test_aconvert_fireworks_stream_lifecycle() -> None:
    async def _araw() -> Any:
        for chunk in _reasoning_text_tool():
            yield chunk

    events: list[Any] = [e async for e in aconvert_fireworks_stream(_araw())]
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "fireworks"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["reasoning", "text", "tool_call"]
    assert (
        cast("dict[str, Any]", finishes[0]["content"])["reasoning"] == "Let me think."
    )
    assert cast("dict[str, Any]", finishes[1]["content"])["text"] == "Hi there"
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def _parallel_tool_calls() -> list[dict]:
    """Two tool calls interleaved at index 0 and 1 across chunks."""
    m = "accounts/fireworks/models/deepseek-r1"
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


def test_convert_fireworks_stream_parallel_tool_calls() -> None:
    """Args at distinct `tool:{idx}` keys finalize as separate tool calls."""
    events: list[Any] = list(convert_fireworks_stream(iter(_parallel_tool_calls())))
    assert_valid_event_stream(events)

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["tool_call", "tool_call"]

    first = cast("dict[str, Any]", finishes[0]["content"])
    second = cast("dict[str, Any]", finishes[1]["content"])
    assert first["name"] == "get_weather"
    assert first["args"] == {"city": "Paris"}
    assert second["name"] == "get_time"
    assert second["args"] == {"zone": "UTC"}

    # No usage chunk in this fixture: message-finish should carry no usage.
    assert events[-1]["event"] == "message-finish"
    assert events[-1].get("usage") is None
