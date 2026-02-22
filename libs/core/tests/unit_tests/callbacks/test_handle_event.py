"""Tests for the `on_chat_model_start` fallback in `handle_event`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from langchain_core.callbacks.manager import handle_event
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.base import AsyncBaseTracer

if TYPE_CHECKING:
    import pytest

    from langchain_core.tracers.schemas import Run

SERIALIZED = {"id": ["chat_model"]}


class _NoOpAsyncTracer(AsyncBaseTracer):
    """Async tracer that does NOT override `on_chat_model_start`.

    Records `on_llm_start` calls so the test can verify the fallback fired.
    """

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []
        self.llm_start_calls: list[dict[str, Any]] = []

    async def _persist_run(self, run: Run) -> None:
        self.runs.append(run)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        self.llm_start_calls.append(
            {
                "serialized": serialized,
                "prompts": prompts,
                "run_id": run_id,
            }
        )


class _WorkingAsyncTracer(AsyncBaseTracer):
    """Async tracer that DOES override `on_chat_model_start`.

    Used to verify the normal (non-fallback) path still works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []
        self.chat_model_start_calls: list[dict[str, Any]] = []

    async def _persist_run(self, run: Run) -> None:
        self.runs.append(run)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **_kwargs: Any,
    ) -> None:
        self.chat_model_start_calls.append(
            {
                "serialized": serialized,
                "messages": messages,
                "run_id": run_id,
            }
        )


def test_async_tracer_falls_back_to_on_llm_start_in_sync_context() -> None:
    """Async tracer without `on_chat_model_start` falls back.

    When `handle_event` is called synchronously with an
    `AsyncBaseTracer` that doesn't implement `on_chat_model_start`,
    the `on_llm_start` callback should fire as a fallback.
    """
    tracer = _NoOpAsyncTracer()
    run_id = uuid4()
    messages = [[SystemMessage(content="sys"), HumanMessage(content="hi")]]

    handle_event(
        [tracer],
        "on_chat_model_start",
        "ignore_chat_model",
        SERIALIZED,
        messages,
        run_id=run_id,
    )

    assert len(tracer.llm_start_calls) == 1
    call = tracer.llm_start_calls[0]
    assert call["serialized"] == SERIALIZED
    # The fallback converts messages to strings via get_buffer_string
    assert isinstance(call["prompts"], list)
    assert len(call["prompts"]) == 1
    assert isinstance(call["prompts"][0], str)


def test_async_tracer_no_fallback_when_implemented() -> None:
    """Async tracer WITH `on_chat_model_start` does not fall back.

    When the handler implements `on_chat_model_start`, no fallback
    should be triggered.
    """
    tracer = _WorkingAsyncTracer()
    run_id = uuid4()
    messages = [[HumanMessage(content="hello")]]

    handle_event(
        [tracer],
        "on_chat_model_start",
        "ignore_chat_model",
        SERIALIZED,
        messages,
        run_id=run_id,
    )

    assert len(tracer.chat_model_start_calls) == 1
    call = tracer.chat_model_start_calls[0]
    assert call["serialized"] == SERIALIZED
    assert call["messages"] == messages


def test_async_tracer_fallback_no_error_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback path should not produce any warning/error logs."""
    tracer = _NoOpAsyncTracer()
    messages = [[HumanMessage(content="test")]]

    with caplog.at_level("WARNING", logger="langchain_core.callbacks.manager"):
        handle_event(
            [tracer],
            "on_chat_model_start",
            "ignore_chat_model",
            SERIALIZED,
            messages,
            run_id=uuid4(),
        )

    assert not caplog.records, (
        f"Expected no warnings but got: {[r.message for r in caplog.records]}"
    )
