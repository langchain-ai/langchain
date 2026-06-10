"""Unit tests for the Ollama native stream-events converter."""

from typing import Any, cast
from unittest.mock import patch

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_ollama import ChatOllama
from langchain_ollama._stream_events import (
    aconvert_ollama_stream,
    convert_ollama_stream,
)
from langchain_ollama.chat_models import _get_tool_calls_from_response


def _thinking_then_text_then_tool() -> list[dict]:
    return [
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "", "thinking": "Let me "},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "", "thinking": "think."},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "The answer"},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": " is 42."},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    }
                ],
            },
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 7,
        },
    ]


def test_convert_ollama_stream_lifecycle() -> None:
    events: list[Any] = list(
        convert_ollama_stream(
            iter(_thinking_then_text_then_tool()),
            _get_tool_calls_from_response,
            reasoning=True,
        )
    )
    assert_valid_event_stream(events)

    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""  # empty → core's seeded run id stands
    assert events[0]["metadata"]["provider"] == "ollama"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["reasoning", "text", "tool_call"]
    reasoning = cast("dict[str, Any]", finishes[0]["content"])
    assert reasoning["reasoning"] == "Let me think."
    text = cast("dict[str, Any]", finishes[1]["content"])
    assert text["text"] == "The answer is 42."
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
    }


async def test_aconvert_ollama_stream_lifecycle() -> None:
    async def _araw() -> Any:
        for chunk in _thinking_then_text_then_tool():
            yield chunk

    events: list[Any] = [
        ev
        async for ev in aconvert_ollama_stream(
            _araw(),
            _get_tool_calls_from_response,
            reasoning=True,
        )
    ]
    assert_valid_event_stream(events)

    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "ollama"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["reasoning", "text", "tool_call"]
    reasoning = cast("dict[str, Any]", finishes[0]["content"])
    assert reasoning["reasoning"] == "Let me think."
    text = cast("dict[str, Any]", finishes[1]["content"])
    assert text["text"] == "The answer is 42."
    tool = cast("dict[str, Any]", finishes[2]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
    }


def test_convert_ollama_stream_reasoning_disabled() -> None:
    """With reasoning off, thinking is not surfaced as a block."""
    chunks = [
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "", "thinking": "hidden"},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "hi"},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    ]
    events: list[Any] = list(
        convert_ollama_stream(
            iter(chunks), _get_tool_calls_from_response, reasoning=False
        )
    )
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text"]


def test_convert_ollama_stream_skips_leading_load_response() -> None:
    """A leading `done_reason="load"` empty chunk is skipped, like the bridge."""
    chunks = [
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "load",
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": "hi"},
            "done": False,
        },
        {
            "model": "qw3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    ]
    events: list[Any] = list(
        convert_ollama_stream(
            iter(chunks), _get_tool_calls_from_response, reasoning=True
        )
    )
    assert_valid_event_stream(events)
    # message-start carries the model from the first *non-load* chunk, not the
    # skipped load chunk, and the stream is not empty.
    assert events[0]["event"] == "message-start"
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text"]


def test_ollama_stream_events_v3_model_level() -> None:
    """End-to-end: `stream_events(version="v3")` drives the native hook."""
    llm = ChatOllama(model="qw3", reasoning=True)
    with patch.object(
        ChatOllama,
        "_create_chat_stream",
        return_value=iter(_thinking_then_text_then_tool()),
    ):
        events: list[Any] = list(llm.stream_events("hi", version="v3"))

    assert_valid_event_stream(events)
    # message-start id is the LC run id (threaded by core), not empty.
    assert events[0]["event"] == "message-start"
    assert events[0]["id"]
    finish_types = [
        e["content"]["type"] for e in events if e["event"] == "content-block-finish"
    ]
    assert finish_types == ["reasoning", "text", "tool_call"]
    assert events[-1]["metadata"]["model_provider"] == "ollama"


async def test_ollama_astream_events_v3_model_level() -> None:
    """Async end-to-end: `astream_events(version="v3")` drives the native hook."""

    async def _araw() -> Any:
        for chunk in _thinking_then_text_then_tool():
            yield chunk

    llm = ChatOllama(model="qw3", reasoning=True)
    with patch.object(ChatOllama, "_acreate_chat_stream", return_value=_araw()):
        stream = await llm.astream_events("hi", version="v3")
        events: list[Any] = [ev async for ev in stream]

    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"]
    finish_types = [
        e["content"]["type"] for e in events if e["event"] == "content-block-finish"
    ]
    assert finish_types == ["reasoning", "text", "tool_call"]
    assert events[-1]["metadata"]["model_provider"] == "ollama"
