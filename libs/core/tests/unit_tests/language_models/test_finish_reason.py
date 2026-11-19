"""Tests for passing provider finish fields through `stream_events`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import Field

from langchain_core.language_models._compat_bridge import (
    chunks_to_events,
    message_to_events,
)
from langchain_core.language_models._finish_reason import finish_fields
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage

_STOP = {"finish_reason": "stop", "normalized_finish_reason": "stop"}


def test_finish_fields_passes_through() -> None:
    metadata = {
        "model_provider": "anthropic",
        "finish_reason": "pause_turn",
        "normalized_finish_reason": "other",
        "finish_state": "suspended",
    }
    assert finish_fields(metadata) == {
        "finish_reason": "pause_turn",
        "normalized_finish_reason": "other",
        "finish_state": "suspended",
    }


def test_finish_fields_empty_without_provider_fields() -> None:
    assert finish_fields(None) == {}
    assert finish_fields({"stop_reason": "end_turn"}) == {}


def test_finish_fields_raw_only() -> None:
    assert finish_fields({"finish_reason": "stop"}) == {"finish_reason": "stop"}
    assert finish_fields({"finish_reason": "stop"}, has_refusal=True) == {
        "finish_reason": "stop"
    }


def test_finish_fields_stop_with_tool_calls_is_tool_call() -> None:
    fields = finish_fields(_STOP, has_tool_calls=True)
    assert fields["normalized_finish_reason"] == "tool_call"
    assert fields["finish_reason"] == "stop"


def test_finish_fields_length_with_tool_calls_is_kept() -> None:
    metadata = {"finish_reason": "length", "normalized_finish_reason": "length"}
    fields = finish_fields(metadata, has_tool_calls=True)
    assert fields["normalized_finish_reason"] == "length"


def test_finish_fields_refusal_overrides() -> None:
    fields = finish_fields(_STOP, has_refusal=True)
    assert fields["normalized_finish_reason"] == "refusal"


def _finish_event(events: list[Any]) -> dict[str, Any]:
    finish = events[-1]
    assert finish["event"] == "message-finish"
    return cast("dict[str, Any]", finish)


def test_chunks_to_events_passes_provider_fields() -> None:
    chunks = [
        ChatGenerationChunk(message=AIMessageChunk(content="hi")),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={
                    "stop_reason": "pause_turn",
                    "finish_reason": "pause_turn",
                    "normalized_finish_reason": "other",
                    "finish_state": "suspended",
                },
            )
        ),
    ]
    finish = _finish_event(list(chunks_to_events(iter(chunks))))
    assert finish["finish_reason"] == "pause_turn"
    assert finish["normalized_finish_reason"] == "other"
    assert finish["finish_state"] == "suspended"
    assert finish["metadata"]["stop_reason"] == "pause_turn"


def test_chunks_to_events_tool_call_override() -> None:
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "type": "tool_call_chunk",
                        "id": "call_1",
                        "name": "search",
                        "args": "{}",
                        "index": 0,
                    }
                ],
                response_metadata=_STOP,
            )
        ),
    ]
    finish = _finish_event(list(chunks_to_events(iter(chunks))))
    assert finish["finish_reason"] == "stop"
    assert finish["normalized_finish_reason"] == "tool_call"
    assert "finish_state" not in finish


def test_chunks_to_events_without_provider_fields() -> None:
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="hi", response_metadata={"stop_reason": "end_turn"}
            )
        )
    ]
    finish = _finish_event(list(chunks_to_events(iter(chunks))))
    assert "finish_reason" not in finish
    assert "normalized_finish_reason" not in finish
    assert "finish_state" not in finish


def test_message_to_events_tool_call_override() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search", "args": {}}],
        response_metadata=_STOP,
    )
    finish = _finish_event(list(message_to_events(msg)))
    assert finish["finish_reason"] == "stop"
    assert finish["normalized_finish_reason"] == "tool_call"


class _StreamModel(BaseChatModel):
    first: dict[str, Any] = Field(default_factory=dict)
    last: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "finish-reason-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = AIMessage(content="hi", response_metadata=dict(self.last))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(message=AIMessageChunk(content="", **self.first))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata=dict(self.last))
        )


_LENGTH = {
    "finish_reason": "max_tokens",
    "normalized_finish_reason": "length",
    "finish_state": "complete",
}


def test_stream_events_v3_carries_finish_fields() -> None:
    stream = _StreamModel(last=_LENGTH).stream_events("hi", version="v3")
    msg = stream.output
    finish = _finish_event(list(stream))
    assert finish["finish_reason"] == "max_tokens"
    assert finish["normalized_finish_reason"] == "length"
    assert finish["finish_state"] == "complete"
    assert msg.response_metadata["finish_reason"] == "max_tokens"
    assert msg.response_metadata["normalized_finish_reason"] == "length"
    assert msg.response_metadata["finish_state"] == "complete"


async def test_astream_events_v3_carries_finish_fields() -> None:
    stream = await _StreamModel(last=_LENGTH).astream_events("hi", version="v3")
    msg = await stream
    events = [event async for event in stream]
    assert _finish_event(events)["normalized_finish_reason"] == "length"
    assert msg.response_metadata["normalized_finish_reason"] == "length"


def test_stream_events_v3_refusal_overrides_stop() -> None:
    model = _StreamModel(
        first={
            "additional_kwargs": {"refusal": "I can't help with that."},
            "response_metadata": {"model_provider": "openai"},
        },
        last=_STOP,
    )
    stream = model.stream_events("hi", version="v3")
    msg = stream.output
    finish = _finish_event(list(stream))
    assert finish["finish_reason"] == "stop"
    assert finish["normalized_finish_reason"] == "refusal"
    assert msg.response_metadata["normalized_finish_reason"] == "refusal"


def test_invoke_leaves_response_metadata_alone() -> None:
    msg = _StreamModel(last={"finish_reason": "stop"}).invoke("hi")
    assert msg.response_metadata["finish_reason"] == "stop"
    assert "normalized_finish_reason" not in msg.response_metadata
    assert "finish_state" not in msg.response_metadata
