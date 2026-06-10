"""Unit tests for the MistralAI native stream-events converter."""

from typing import Any, cast
from unittest.mock import patch

from langchain_tests.utils.stream_lifecycle import assert_valid_event_stream

from langchain_mistralai import ChatMistralAI
from langchain_mistralai._stream_events import convert_mistral_stream
from langchain_mistralai.chat_models import _convert_chunk_to_message_chunk


def _text_then_tool() -> list[dict]:
    cid, model = "cmpl-1", "mistral-large"
    return [
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " world"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "t1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
            },
        },
    ]


def _reasoning_then_text_v1() -> list[dict]:
    """Mistral ``output_version="v1"`` chunks: a thinking block then text.

    Under v1 `delta.content` is a list of typed blocks. A `thinking` block
    carries its text in a `thinking` sub-block list; `_convert_chunk_to_message_chunk`
    maps it to a `reasoning` content block. When the block `type` changes
    (`thinking` -> `text`) the converter's threaded `index`/`index_type`
    advance, splitting the stream into two distinct blocks.
    """
    cid, model = "cmpl-1", "magistral-medium"
    return [
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "Let me "}],
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "think."}],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": cid,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": [{"type": "text", "text": "Hi Bob."}]},
                    "finish_reason": "stop",
                }
            ],
        },
    ]


def test_convert_mistral_stream_v1_reasoning() -> None:
    """v1 reasoning path: index/index_type threading splits thinking from text.

    Guards the bespoke converter's core motivation — that the
    `index`/`index_type` returned by `_convert_chunk_to_message_chunk` are
    threaded back in so a type change (`thinking` -> `text`) opens a new
    block rather than merging. The reasoning-as-blocks behavior against a
    live model is covered by `test_reasoning_v1` in the integration tests.
    """
    events: list[Any] = list(
        convert_mistral_stream(
            iter(_reasoning_then_text_v1()),
            _convert_chunk_to_message_chunk,
            output_version="v1",
        )
    )
    assert_valid_event_stream(events)

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    # Two distinct blocks: the thinking deltas accumulate into one reasoning
    # block, then the type change to `text` advances index/index_type.
    assert [f["content"]["type"] for f in finishes] == ["reasoning", "text"]
    assert [f["index"] for f in finishes] == [0, 1]

    reasoning = cast("dict[str, Any]", finishes[0]["content"])
    text = cast("dict[str, Any]", finishes[1]["content"])
    assert reasoning["reasoning"] == "Let me think."
    assert text["text"] == "Hi Bob."


def test_convert_mistral_stream_lifecycle() -> None:
    events: list[Any] = list(
        convert_mistral_stream(
            iter(_text_then_tool()),
            _convert_chunk_to_message_chunk,
            output_version="v0",
        )
    )
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["id"] == ""
    assert events[0]["metadata"]["provider"] == "mistralai"

    text = "".join(
        e["delta"].get("text", "")
        for e in events
        if e["event"] == "content-block-delta"
        and e["delta"].get("type") == "text-delta"
    )
    assert text == "Hello world"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tool_finishes = [f for f in finishes if f["content"]["type"] == "tool_call"]
    assert len(tool_finishes) == 1
    tc = cast("dict[str, Any]", tool_finishes[0]["content"])
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
    }


def test_mistral_stream_events_v3_lifecycle() -> None:
    """Validate `stream_events(version="v3")` over a text + tool_call stream.

    Threads a realistic chunk sequence through `_stream_chat_model_events`
    via a mocked raw client and asserts a spec-conformant event stream.
    """
    llm = ChatMistralAI(api_key="test")  # type: ignore[arg-type]

    with patch.object(
        ChatMistralAI,
        "completion_with_retry",
        return_value=iter(_text_then_tool()),
    ):
        events: list[Any] = list(llm.stream_events("Test query", version="v3"))

    assert_valid_event_stream(events)

    # `message-start` must carry the stream's LangChain run id (threaded from
    # core), not the empty converter default.
    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tool_finishes = [f for f in finishes if f["content"]["type"] == "tool_call"]
    assert len(tool_finishes) == 1
    tc = cast("dict[str, Any]", tool_finishes[0]["content"])
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Paris"}

    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "mistralai"


async def test_mistral_astream_events_v3_lifecycle() -> None:
    """Async twin of `test_mistral_stream_events_v3_lifecycle`."""
    llm = ChatMistralAI(api_key="test")  # type: ignore[arg-type]

    async def _acompletion(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> Any:
            for chunk in _text_then_tool():
                yield chunk

        return _gen()

    with patch(
        "langchain_mistralai.chat_models.acompletion_with_retry",
        new=_acompletion,
    ):
        stream = await llm.astream_events("Test query", version="v3")
        events: list[Any] = [e async for e in stream]

    assert_valid_event_stream(events)
    message_start = cast("dict[str, Any]", events[0])
    assert message_start["event"] == "message-start"
    assert message_start["id"]

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tool_finishes = [f for f in finishes if f["content"]["type"] == "tool_call"]
    assert len(tool_finishes) == 1
    tc = cast("dict[str, Any]", tool_finishes[0]["content"])
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Paris"}

    message_finish = cast("dict[str, Any]", events[-1])
    assert message_finish["event"] == "message-finish"
    assert message_finish["metadata"]["model_provider"] == "mistralai"


async def test_aconvert_mistral_stream_lifecycle() -> None:
    from langchain_mistralai._stream_events import aconvert_mistral_stream

    async def _araw() -> Any:
        for chunk in _text_then_tool():
            yield chunk

    events: list[Any] = [
        e
        async for e in aconvert_mistral_stream(
            _araw(), _convert_chunk_to_message_chunk, output_version="v0"
        )
    ]
    assert_valid_event_stream(events)
    assert events[0]["event"] == "message-start"
    assert events[0]["metadata"]["provider"] == "mistralai"

    finishes = [e for e in events if e["event"] == "content-block-finish"]
    tool_finishes = [f for f in finishes if f["content"]["type"] == "tool_call"]
    assert len(tool_finishes) == 1
    tc = cast("dict[str, Any]", tool_finishes[0]["content"])
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"city": "Paris"}

    message_finish = events[-1]
    assert message_finish["event"] == "message-finish"
    assert message_finish["usage"] == {
        "input_tokens": 9,
        "output_tokens": 3,
        "total_tokens": 12,
    }
