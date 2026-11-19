"""Tests for finish-reason normalization and its `stream_events` wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import Field

from langchain_core.language_models._compat_bridge import (
    chunks_to_events,
    message_to_events,
)
from langchain_core.language_models._finish_reason import (
    get_raw_finish_reason,
    normalize_finish_reason,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage


@pytest.mark.parametrize(
    ("provider", "raw", "expected"),
    [
        ("openai", "stop", "stop"),
        ("openai", "length", "length"),
        ("openai", "tool_calls", "tool_call"),
        ("openai", "function_call", "tool_call"),
        ("openai", "content_filter", "content_filter"),
        ("anthropic", "end_turn", "stop"),
        ("anthropic", "stop_sequence", "stop"),
        ("anthropic", "max_tokens", "length"),
        ("anthropic", "tool_use", "tool_call"),
        ("anthropic", "refusal", "refusal"),
        ("anthropic", "model_context_window_exceeded", "context_exceeded"),
        ("google_genai", "STOP", "stop"),
        ("google_genai", "MAX_TOKENS", "length"),
        ("google_genai", "SAFETY", "refusal"),
        ("google_genai", "RECITATION", "content_filter"),
        ("google_genai", "BLOCKLIST", "refusal"),
        ("google_genai", "PROHIBITED_CONTENT", "refusal"),
        ("google_genai", "SPII", "content_filter"),
        ("google_genai", "LANGUAGE", "content_filter"),
        ("google_genai", "IMAGE_SAFETY", "content_filter"),
        ("google_genai", "IMAGE_PROHIBITED_CONTENT", "content_filter"),
        ("google_genai", "IMAGE_RECITATION", "content_filter"),
        ("google_genai", "MALFORMED_FUNCTION_CALL", "other"),
        ("google_genai", "OTHER", "other"),
        ("google_vertexai", "stop", "stop"),
        ("google_vertexai", "max_tokens", "length"),
        (None, "stop", "stop"),
        (None, "length", "length"),
        (None, "tool_calls", "tool_call"),
        ("some_provider", "content_filter", "content_filter"),
    ],
)
def test_normalize_finish_reason_mapping(
    provider: str | None, raw: str, expected: str
) -> None:
    metadata: dict[str, Any] = {"finish_reason": raw}
    if provider is not None:
        metadata["model_provider"] = provider
    assert normalize_finish_reason(metadata) == (expected, "complete")


@pytest.mark.parametrize(
    ("provider", "raw"),
    [
        ("openai", "something_new"),
        ("anthropic", "something_new"),
        ("google_genai", "FINISH_REASON_UNSPECIFIED"),
        ("some_provider", "weird"),
        (None, "weird"),
    ],
)
def test_unknown_finish_reason_maps_to_other(provider: str | None, raw: str) -> None:
    metadata: dict[str, Any] = {"finish_reason": raw}
    if provider is not None:
        metadata["model_provider"] = provider
    assert normalize_finish_reason(metadata) == ("other", "complete")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "stop"),
        ("tool_use", "tool_call"),
        ("guardrail_intervened", "content_filter"),
        ("MAX_TOKENS", "length"),
        ("IMAGE-SAFETY", "content_filter"),
    ],
)
def test_unknown_provider_uses_generic_table(raw: str, expected: str) -> None:
    metadata = {"model_provider": "bedrock_converse", "stop_reason": raw}
    assert normalize_finish_reason(metadata) == (expected, "complete")


def test_openai_responses_raw_reason_is_status() -> None:
    metadata = {"model_provider": "openai", "status": "incomplete"}
    assert get_raw_finish_reason(metadata) == "incomplete"


def test_anthropic_reads_stop_reason_key() -> None:
    metadata = {"model_provider": "anthropic", "stop_reason": "max_tokens"}
    assert normalize_finish_reason(metadata) == ("length", "complete")


def test_pause_turn_is_suspended() -> None:
    metadata = {"model_provider": "anthropic", "stop_reason": "pause_turn"}
    assert normalize_finish_reason(metadata) == ("other", "suspended")


def test_missing_finish_reason_is_none() -> None:
    assert normalize_finish_reason({"model_provider": "anthropic"}) == (None, None)
    assert normalize_finish_reason(None) == (None, None)


@pytest.mark.parametrize(
    ("provider", "raw"),
    [
        ("openai", "stop"),
        ("anthropic", "end_turn"),
        ("google_genai", "STOP"),
        (None, "stop"),
    ],
)
def test_stop_with_tool_calls_is_tool_call(provider: str | None, raw: str) -> None:
    metadata: dict[str, Any] = {"finish_reason": raw}
    if provider is not None:
        metadata["model_provider"] = provider
    assert normalize_finish_reason(metadata, has_tool_calls=True) == (
        "tool_call",
        "complete",
    )


def test_non_stop_reason_not_overridden_by_tool_calls() -> None:
    metadata = {"model_provider": "openai", "finish_reason": "length"}
    assert normalize_finish_reason(metadata, has_tool_calls=True) == (
        "length",
        "complete",
    )


@pytest.mark.parametrize(
    ("metadata", "has_tool_calls", "expected"),
    [
        ({"status": "completed"}, False, "stop"),
        ({"status": "completed"}, True, "tool_call"),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            False,
            "length",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            False,
            "content_filter",
        ),
        ({"status": "incomplete", "incomplete_details": None}, False, "other"),
        ({"status": "failed"}, False, "error"),
        ({"status": "cancelled"}, False, "other"),
    ],
)
def test_openai_responses_status(
    metadata: dict[str, Any], *, has_tool_calls: bool, expected: str
) -> None:
    metadata = {**metadata, "model_provider": "openai"}
    assert normalize_finish_reason(metadata, has_tool_calls=has_tool_calls) == (
        expected,
        "complete",
    )


def test_status_ignored_for_non_openai_provider() -> None:
    metadata = {"model_provider": "anthropic", "status": "completed"}
    assert normalize_finish_reason(metadata) == (None, None)


def _finish_event(events: list[Any]) -> dict[str, Any]:
    finish = events[-1]
    assert finish["event"] == "message-finish"
    return cast("dict[str, Any]", finish)


def test_chunks_to_events_sets_finish_fields() -> None:
    chunks = [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="hi", response_metadata={"model_provider": "anthropic"}
            )
        ),
        ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"stop_reason": "pause_turn"},
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
                response_metadata={
                    "model_provider": "google_genai",
                    "finish_reason": "STOP",
                },
            )
        ),
    ]
    finish = _finish_event(list(chunks_to_events(iter(chunks))))
    assert finish["finish_reason"] == "STOP"
    assert finish["normalized_finish_reason"] == "tool_call"
    assert finish["finish_state"] == "complete"


def test_chunks_to_events_omits_normalized_without_raw() -> None:
    chunks = [ChatGenerationChunk(message=AIMessageChunk(content="hi"))]
    finish = _finish_event(list(chunks_to_events(iter(chunks))))
    assert "finish_reason" not in finish
    assert "normalized_finish_reason" not in finish
    assert "finish_state" not in finish


def test_message_to_events_sets_finish_fields() -> None:
    msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search", "args": {}}],
        response_metadata={"model_provider": "openai", "finish_reason": "stop"},
    )
    finish = _finish_event(list(message_to_events(msg)))
    assert finish["finish_reason"] == "stop"
    assert finish["normalized_finish_reason"] == "tool_call"
    assert finish["finish_state"] == "complete"


class _FinishReasonModel(BaseChatModel):
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
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="hi", response_metadata={"model_provider": "anthropic"}
            )
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata={"stop_reason": "max_tokens"}
            )
        )


def test_stream_events_v3_carries_finish_fields() -> None:
    stream = _FinishReasonModel().stream_events("hi", version="v3")
    msg = stream.output
    finish = _finish_event(list(stream))
    assert finish["finish_reason"] == "max_tokens"
    assert finish["normalized_finish_reason"] == "length"
    assert finish["finish_state"] == "complete"
    assert msg.response_metadata["stop_reason"] == "max_tokens"
    assert msg.response_metadata["finish_reason"] == "max_tokens"
    assert msg.response_metadata["normalized_finish_reason"] == "length"
    assert msg.response_metadata["finish_state"] == "complete"


async def test_astream_events_v3_carries_finish_fields() -> None:
    stream = await _FinishReasonModel().astream_events("hi", version="v3")
    msg = await stream
    events = [event async for event in stream]
    finish = _finish_event(events)
    assert finish["normalized_finish_reason"] == "length"
    assert msg.response_metadata["normalized_finish_reason"] == "length"
    assert msg.response_metadata["finish_state"] == "complete"


class _InvokeModel(BaseChatModel):
    response_metadata: dict[str, Any]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    refusal: str | None = None

    @property
    def _llm_type(self) -> str:
        return "invoke-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        message = AIMessage(
            content="hi",
            additional_kwargs={"refusal": self.refusal} if self.refusal else {},
            response_metadata=dict(self.response_metadata),
            tool_calls=cast("Any", self.tool_calls),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_invoke_adds_finish_fields() -> None:
    msg = _InvokeModel(
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"}
    ).invoke("hi")
    assert msg.response_metadata["stop_reason"] == "refusal"
    assert msg.response_metadata["finish_reason"] == "refusal"
    assert msg.response_metadata["normalized_finish_reason"] == "refusal"
    assert msg.response_metadata["finish_state"] == "complete"


async def test_ainvoke_adds_finish_fields() -> None:
    msg = await _InvokeModel(
        response_metadata={"model_provider": "anthropic", "stop_reason": "pause_turn"}
    ).ainvoke("hi")
    assert msg.response_metadata["normalized_finish_reason"] == "other"
    assert msg.response_metadata["finish_state"] == "suspended"


def test_invoke_keeps_provider_finish_reason() -> None:
    msg = _InvokeModel(
        response_metadata={"model_provider": "google_genai", "finish_reason": "STOP"},
        tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
    ).invoke("hi")
    assert msg.response_metadata["finish_reason"] == "STOP"
    assert msg.response_metadata["normalized_finish_reason"] == "tool_call"


def test_invoke_without_finish_reason_adds_nothing() -> None:
    msg = _InvokeModel(response_metadata={"model_provider": "anthropic"}).invoke("hi")
    assert "finish_reason" not in msg.response_metadata
    assert "normalized_finish_reason" not in msg.response_metadata
    assert "finish_state" not in msg.response_metadata


def test_normalize_refusal_overrides_stop() -> None:
    assert normalize_finish_reason(
        {"model_provider": "openai", "finish_reason": "stop"}, has_refusal=True
    ) == ("refusal", "complete")


def test_invoke_openai_refusal_is_refusal() -> None:
    model = _InvokeModel(
        response_metadata={"model_provider": "openai", "finish_reason": "stop"}
    )
    model.refusal = "I can't help with that."
    msg = model.invoke("hi")
    assert msg.response_metadata["finish_reason"] == "stop"
    assert msg.response_metadata["normalized_finish_reason"] == "refusal"


class _RefusalStreamModel(_FinishReasonModel):
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                additional_kwargs={"refusal": "I can't help with that."},
                response_metadata={"model_provider": "openai"},
            )
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata={"finish_reason": "stop"}
            )
        )


def test_stream_events_v3_openai_refusal_is_refusal() -> None:
    stream = _RefusalStreamModel().stream_events("hi", version="v3")
    msg = stream.output
    finish = _finish_event(list(stream))
    assert finish["finish_reason"] == "stop"
    assert finish["normalized_finish_reason"] == "refusal"
    assert msg.response_metadata["normalized_finish_reason"] == "refusal"
