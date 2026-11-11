from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langsmith import Client

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import tracing_error_handler, tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer
from tests.unit_tests.tracers.test_async_base_tracer import FakeAsyncTracer
from tests.unit_tests.tracers.test_base_tracer import FakeTracer

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tracers.schemas import Run


class MetadataError(BaseException):
    code = "custom_failure"


@pytest.mark.parametrize("tracer_class", [FakeTracer, FakeAsyncTracer])
@pytest.mark.parametrize(
    "run_type", ["chain", "llm", "chat_model", "tool", "retriever"]
)
async def test_error_handler_before_persistence(
    tracer_class: type[FakeTracer | FakeAsyncTracer], run_type: str
) -> None:
    tracer = tracer_class()
    error = MetadataError("failed")
    seen: list[tuple[BaseException, Run, str | None, bool, bool]] = []

    def handler(original: BaseException, run: Run) -> None:
        seen.append((original, run, run.error, bool(run.end_time), bool(tracer.runs)))
        run.extra.setdefault("metadata", {})["error_code"] = error.code
        run.error = "customized error"

    manager = AsyncCallbackManager(handlers=[tracer])
    inputs: Any = {"chain": {}, "llm": ["prompt"], "chat_model": [[]]}.get(
        run_type, "query"
    )
    with tracing_error_handler(handler):
        started = await getattr(manager, f"on_{run_type}_start")(
            {"name": run_type}, inputs
        )
        run_manager = started[0] if isinstance(started, list) else started
        error_type = "llm" if run_type == "chat_model" else run_type
        await getattr(run_manager, f"on_{error_type}_error")(error)

    assert len(seen) == 1
    original, live_run, default_error, ended, persisted = seen[0]
    assert original is error
    assert default_error
    assert "MetadataError" in default_error
    assert ended
    assert not persisted
    assert tracer.runs[0] is live_run
    assert live_run.events[-1]["name"] == "error"
    assert live_run.error == "customized error"
    assert live_run.extra["metadata"]["error_code"] == error.code
    assert not tracer.run_map


def _fail(tracer: FakeTracer) -> Run:
    run_id = uuid4()
    tracer.on_chain_start({"name": "chain"}, {}, run_id=run_id)
    return tracer.on_chain_error(ValueError("failed"), run_id=run_id)


def test_nested_handlers_restore_after_exception() -> None:
    tracer = FakeTracer()
    calls: list[str] = []

    def outer(_error: BaseException, run: Run) -> None:
        calls.append("outer")
        run.error = "outer"

    def inner(_error: BaseException, run: Run) -> None:
        calls.append(str(run.error))
        run.error = "inner"

    def fail_in_context() -> None:
        with tracing_error_handler(inner):
            assert _fail(tracer).error == "inner"
            msg = "exit"
            raise ValueError(msg)

    with tracing_error_handler(outer):
        with pytest.raises(ValueError, match="exit"):
            fail_in_context()
        assert _fail(tracer).error == "outer"
    assert "ValueError" in str(_fail(tracer).error)
    assert calls == ["outer", "outer", "outer"]


def test_handler_failure_does_not_prevent_persistence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = FakeTracer()
    broken = MagicMock(side_effect=ValueError("handler failed"))
    following = MagicMock()
    with tracing_error_handler(broken), tracing_error_handler(following):
        run = _fail(tracer)
    broken.assert_called_once()
    following.assert_called_once_with(broken.call_args.args[0], run)
    assert tracer.runs == [run]
    assert not tracer.run_map
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Error in tracing error handler"
    )
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None
    assert record.exc_info[1] is broken.side_effect
    assert "ValueError: handler failed" in caplog.text


async def test_handler_task_isolation_and_inheritance() -> None:
    tracer = FakeTracer()
    handlers = [MagicMock(), MagicMock()]

    async def isolated(handler: Callable[[BaseException, Run], None]) -> None:
        with tracing_error_handler(handler):
            await asyncio.sleep(0)
            await asyncio.create_task(asyncio.to_thread(_fail, tracer))

    await asyncio.gather(*(isolated(handler) for handler in handlers))
    _fail(tracer)
    assert [handler.call_count for handler in handlers] == [1, 1]
    assert handlers[0].call_args.args[1] is not handlers[1].call_args.args[1]


def test_partial_llm_response_and_success() -> None:
    tracer = FakeTracer()
    handler = MagicMock()
    response = LLMResult(generations=[[]])
    with tracing_error_handler(handler):
        run_id = uuid4()
        tracer.on_llm_start({"name": "llm"}, [], run_id=run_id)
        tracer.on_llm_end(response, run_id=run_id)
        handler.assert_not_called()
        run_id = uuid4()
        tracer.on_llm_start({"name": "llm"}, [], run_id=run_id)
        tracer.on_llm_error(ValueError("failed"), run_id=run_id, response=response)
    assert handler.call_args.args[1].outputs == response.model_dump()


class CustomLangChainTracer(LangChainTracer):
    def _on_chain_error(self, run: Run) -> None:
        run.extra.setdefault("metadata", {})["custom_tracer"] = True
        super()._on_chain_error(run)


@pytest.mark.parametrize("tracer_class", [LangChainTracer, CustomLangChainTracer])
@pytest.mark.parametrize("configured", [False, True])
def test_existing_and_context_configured_langchain_tracer(
    tracer_class: type[LangChainTracer], *, configured: bool
) -> None:
    client = MagicMock(spec=Client)
    existing = tracer_class(client=client)
    handler = MagicMock(
        side_effect=lambda _error, run: setattr(run, "error", "handled")
    )
    with (
        tracing_error_handler(handler),
        tracing_v2_enabled(client=client) as contextual,
    ):
        manager = CallbackManager.configure(
            inheritable_callbacks=None if configured else [existing]
        )
        tracer = contextual if configured else existing
        assert tracer in manager.handlers
        run_manager = manager.on_chain_start({"name": "chain"}, {})
        live_run = tracer.run_map[str(run_manager.run_id)]
        run_manager.on_chain_error(ValueError("failed"))
    handler.assert_called_once()
    assert handler.call_args.args[1] is live_run
    assert type(tracer) is (LangChainTracer if configured else tracer_class)
    assert client.update_run.call_args.kwargs["error"] == "handled"
    if not configured and tracer_class is CustomLangChainTracer:
        assert live_run.extra["metadata"]["custom_tracer"] is True
    handler.reset_mock()
    manager.on_chain_start({"name": "chain"}, {}).on_chain_error(ValueError("failed"))
    handler.assert_not_called()
