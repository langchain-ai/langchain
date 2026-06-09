"""Unit tests for the OpenAI Chat Completions native stream-events converter."""

from typing import Any, cast

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream
from pydantic import SecretStr

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._stream_events import (
    convert_openai_completions_stream,
)


def _text_chunks() -> list[dict]:
    cid = "chatcmpl-1"
    return [
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        },
    ]


def _tool_chunks() -> list[dict]:
    cid = "chatcmpl-2"
    return [
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city":'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": ' "Paris"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        },
        {
            "id": cid,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": None,
        },
    ]


def test_convert_openai_completions_text_lifecycle() -> None:
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr("test"))
    events: list[Any] = list(
        convert_openai_completions_stream(
            iter(_text_chunks()), llm._convert_chunk_to_generation_chunk
        )
    )
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    # The provider completion id is deliberately NOT used as the message-start
    # id: on the v3 path core seeds the stream with the LangChain run id, and an
    # empty id here lets that stand (matching the compat bridge).
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "openai"
    text = "".join(
        e["delta"].get("text", "")
        for e in events
        if e["event"] == "content-block-delta"
        and e["delta"].get("type") == "text-delta"
    )
    assert text == "Hello world"
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    assert [f["content"]["type"] for f in finishes] == ["text"]
    assert cast("dict[str, Any]", finishes[0]["content"])["text"] == "Hello world"
    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 7,
        "output_tokens": 2,
        "total_tokens": 9,
    }
    assert message_finish["metadata"]["finish_reason"] == "stop"


def test_convert_openai_completions_tool_call() -> None:
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr("test"))
    events: list[Any] = list(
        convert_openai_completions_stream(
            iter(_tool_chunks()), llm._convert_chunk_to_generation_chunk
        )
    )
    assert_valid_event_stream(events)
    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tool_finishes = [f for f in finishes if f["content"]["type"] == "tool_call"]
    assert len(tool_finishes) == 1
    tc = cast("dict[str, Any]", tool_finishes[0]["content"])
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Paris"}


def test_explicit_message_id_is_used() -> None:
    """An explicit `message_id` (e.g. the v3 stream's run id) is honored."""
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr("test"))
    events: list[Any] = list(
        convert_openai_completions_stream(
            iter(_text_chunks()),
            llm._convert_chunk_to_generation_chunk,
            message_id="lc_run--abc",
        )
    )
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == "lc_run--abc"


def test_provider_override_is_not_clobbered_by_openai() -> None:
    """Reuse with a non-OpenAI `provider` must not be relabeled `openai`.

    `_convert_chunk_to_generation_chunk` hardcodes `model_provider="openai"`;
    the converter must re-apply the caller's `provider` so OpenAI-compatible
    providers (groq, deepseek, ...) stay correctly labeled on both
    `message-start` and `message-finish`.
    """
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr("test"))
    events: list[Any] = list(
        convert_openai_completions_stream(
            iter(_text_chunks()),
            llm._convert_chunk_to_generation_chunk,
            provider="groq",
        )
    )
    assert events[0]["metadata"]["provider"] == "groq"
    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "groq"


async def test_aconvert_openai_completions_text_lifecycle() -> None:
    llm = ChatOpenAI(model="gpt-4o", api_key=SecretStr("test"))

    async def _araw() -> Any:
        for c in _text_chunks():
            yield c

    from langchain_openai.chat_models._stream_events import (
        aconvert_openai_completions_stream,
    )

    events: list[Any] = [
        e
        async for e in aconvert_openai_completions_stream(
            _araw(), llm._convert_chunk_to_generation_chunk
        )
    ]
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[-1]["event"] == "message-finish"
    text = "".join(
        e["delta"].get("text", "")
        for e in events
        if e["event"] == "content-block-delta"
        and e["delta"].get("type") == "text-delta"
    )
    assert text == "Hello world"


def test_response_format_in_model_kwargs_takes_bridge_path() -> None:
    """`response_format` baked into `model_kwargs` must defer to the bridge.

    The native completions converter lacks the beta structured-output
    `get_final_completion()` flow, so such models must route through
    `_stream` (the bridge), never the native converter.
    """
    from unittest.mock import patch

    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    from langchain_openai.chat_models import base

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=SecretStr("test"),
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        yield ChatGenerationChunk(message=AIMessageChunk(content="hi"))

    def _boom(*args: Any, **kwargs: Any) -> Any:
        msg = "native converter must not run for model_kwargs response_format"
        raise AssertionError(msg)

    with (
        patch.object(llm, "_stream", _fake_stream),
        patch.object(base, "convert_openai_completions_stream", _boom),
    ):
        events: list[Any] = list(
            llm._stream_chat_model_events([], stop=None, run_manager=None)
        )

    # Bridge path produced events from the patched `_stream` without ever
    # entering the native converter (which would have raised).
    assert events
    assert events[0]["event"] == "message-start"
    assert events[-1]["event"] == "message-finish"
