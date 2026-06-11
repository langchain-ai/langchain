"""Unit tests for the Perplexity native stream-events converter."""

import os
from typing import Any, cast
from unittest.mock import MagicMock, patch

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_perplexity import ChatPerplexity
from langchain_perplexity._stream_events import (
    aconvert_perplexity_stream,
    convert_perplexity_stream,
)

if "PPLX_API_KEY" not in os.environ:
    os.environ["PPLX_API_KEY"] = "fake-key"


def _text_with_citations() -> list[dict]:
    """Fixture: plain text response with citations and search_results.

    Usage is cumulative — each chunk's `usage` is a running total.
    Includes a final `choices: []` chunk carrying only the cumulative total.
    """
    m = "sonar"
    return [
        {
            "model": m,
            "citations": ["https://example.com/1", "https://example.com/2"],
            "search_results": [{"title": "Example", "url": "https://example.com/1"}],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello "},
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "world"},
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        {
            "model": m,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "total_tokens": 16,
            },
        },
        # Usage-only chunk (choices: []) — final cumulative total. Strictly
        # larger than the prior chunk so the test proves message-finish uses
        # THIS total (last-wins), not the last choices-chunk's total and not a
        # sum of per-chunk deltas.
        {
            "model": m,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
                "num_search_queries": 2,
                "search_context_size": "high",
            },
        },
    ]


def _tool_call_chunks() -> list[dict]:
    """Fixture: single tool call streamed across two chunks."""
    m = "sonar"
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
                                "id": "call_abc",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18},
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
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        },
    ]


def test_convert_perplexity_stream_lifecycle() -> None:
    events: list[Any] = list(convert_perplexity_stream(iter(_text_with_citations())))
    assert_valid_event_stream(events)

    # message-start: empty id (LangChain run id slot), correct provider
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "perplexity"

    # Block order: exactly one text block
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["text"]
    assert cast("dict[str, Any]", finishes[0]["content"])["text"] == "Hello world!"

    # message-finish: usage == LAST cumulative total (the no-choices final
    # chunk's 7/17, not the prior choices-chunk's 6/16 nor a sum of deltas)
    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
    }

    # Extras present in response_metadata
    assert message_finish["metadata"]["model_provider"] == "perplexity"
    assert message_finish["metadata"]["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert message_finish["metadata"]["search_results"] == [
        {"title": "Example", "url": "https://example.com/1"}
    ]


async def test_aconvert_perplexity_stream_lifecycle() -> None:
    async def _araw() -> Any:
        for chunk in _text_with_citations():
            yield chunk

    events: list[Any] = [e async for e in aconvert_perplexity_stream(_araw())]
    assert_valid_event_stream(events)

    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "perplexity"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["text"]
    assert cast("dict[str, Any]", finishes[0]["content"])["text"] == "Hello world!"

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    # Usage must equal the LAST cumulative total (7/17), not a sum of deltas
    assert message_finish["usage"] == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
    }
    assert message_finish["metadata"]["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert message_finish["metadata"]["search_results"] == [
        {"title": "Example", "url": "https://example.com/1"}
    ]


def test_convert_perplexity_stream_tool_call() -> None:
    """Tool calls are surfaced as `tool_call` blocks keyed `tool:{idx}`."""
    events: list[Any] = list(convert_perplexity_stream(iter(_tool_call_chunks())))
    assert_valid_event_stream(events)

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    types = [f["content"]["type"] for f in finishes]
    assert types == ["tool_call"]

    tool = cast("dict[str, Any]", finishes[0]["content"])
    assert tool["name"] == "get_weather"
    assert tool["args"] == {"city": "Paris"}


def test_perplexity_stream_events_v3_lifecycle() -> None:
    """Drive `stream_events(version="v3")` through the native sync hook.

    Confirms the model-level path threads the LangChain run id onto
    `message-start` (non-empty, unlike the converter's empty default) and
    surfaces text blocks with `model_provider == "perplexity"`.
    """
    llm = ChatPerplexity(model="sonar")
    mock_client = MagicMock()

    def mock_create(*_a: Any, **_k: Any) -> Any:
        return iter(_text_with_citations())

    mock_client.chat.completions.create = mock_create

    with patch.object(llm, "client", mock_client):
        stream = llm.stream_events("Test query", version="v3")
        events = list(stream)

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]  # core seeds a non-empty LangChain run id
    assert message_start["metadata"]["provider"] == "perplexity"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text"]

    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "perplexity"
    # Citations ride on message-finish metadata (not additional_kwargs — the v3
    # path has no additional_kwargs channel; this is an intentional relocation).
    assert message_finish["metadata"]["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]

    # Round-trip: extras and model_name reach the assembled message's
    # response_metadata (the user-facing guarantee of the de-risk design).
    output = stream.output
    assert output.response_metadata["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert output.response_metadata["model_name"] == "sonar"


async def test_perplexity_astream_events_v3_lifecycle() -> None:
    """Async twin of `test_perplexity_stream_events_v3_lifecycle`."""
    llm = ChatPerplexity(model="sonar")

    async def _araw() -> Any:
        for chunk in _text_with_citations():
            yield chunk

    async def mock_create(*_a: Any, **_k: Any) -> Any:
        return _araw()

    mock_async_client = MagicMock()
    mock_async_client.chat.completions.create = mock_create

    with patch.object(llm, "async_client", mock_async_client):
        stream = await llm.astream_events("Test query", version="v3")
        events = [e async for e in stream]

    assert_valid_event_stream(events)

    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]  # non-empty LC run id
    assert message_start["metadata"]["provider"] == "perplexity"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text"]

    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "perplexity"
    # Citations ride on message-finish metadata (not additional_kwargs — the v3
    # path has no additional_kwargs channel; this is an intentional relocation).
    assert message_finish["metadata"]["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]

    output = await stream.output
    assert output.response_metadata["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert output.response_metadata["model_name"] == "sonar"


def _no_usage_text() -> list[dict]:
    """Fixture: text chunks with no `usage` field at all."""
    m = "sonar"
    return [
        {
            "model": m,
            "choices": [
                {"index": 0, "delta": {"content": "Hi"}, "finish_reason": "stop"}
            ],
        }
    ]


def test_convert_perplexity_stream_no_usage() -> None:
    """No `usage` on any chunk → message-finish omits the usage field."""
    events: list[Any] = list(convert_perplexity_stream(iter(_no_usage_text())))
    assert_valid_event_stream(events)
    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish.get("usage") is None


def test_convert_perplexity_stream_usage_only_yields_nothing() -> None:
    """A stream with only `choices: []` usage chunks yields no events."""
    usage_only = [
        {"model": "sonar", "choices": [], "usage": {"total_tokens": 5}},
        {"model": "sonar", "choices": [], "usage": {"total_tokens": 9}},
    ]
    assert list(convert_perplexity_stream(iter(usage_only))) == []


def test_convert_perplexity_stream_accepts_sdk_objects() -> None:
    """Non-dict chunks are normalized via `model_dump()`."""

    class _Chunk:
        def __init__(self, data: dict) -> None:
            self._data = data

        def model_dump(self) -> dict:
            return self._data

    raw = [_Chunk(c) for c in _text_with_citations()]
    events: list[Any] = list(convert_perplexity_stream(iter(raw)))
    assert_valid_event_stream(events)
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert cast("dict[str, Any]", finishes[0]["content"])["text"] == "Hello world!"
    assert events[-1]["metadata"]["citations"] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
