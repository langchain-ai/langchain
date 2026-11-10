"""Tests for the explicit run status `LangChainTracer` sends with an error."""

import asyncio
import unittest.mock
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langsmith import Client
from langsmith.run_trees import RunTree

from langchain_core.exceptions import ExpectedCancellation
from langchain_core.tracers.langchain import LangChainTracer


class _StatusRun(RunTree):
    """Stand-in for a `RunTree` from a langsmith version that supports `status`."""

    status: str | None = None


class _OldRun:
    """Stand-in for a `RunTree` from a langsmith version without `status`."""

    model_fields: dict[str, Any] = {}  # noqa: RUF012

    def __init__(self) -> None:
        self.error: str | None = None


class GraphBubbleUp(Exception):  # noqa: N818
    """Same class name as langgraph's, which core matches by name."""


class GraphInterrupt(GraphBubbleUp):
    """Subclass, as in langgraph."""


def _tracer() -> LangChainTracer:
    return LangChainTracer(client=unittest.mock.MagicMock(spec=Client))


def _new_run() -> _StatusRun:
    return _StatusRun(name="x", run_type="chain")


async def _cancel_error(msg: object) -> asyncio.CancelledError:
    task = asyncio.create_task(asyncio.sleep(10))
    await asyncio.sleep(0)
    task.cancel(msg=msg)
    try:
        await task
    except asyncio.CancelledError as e:
        return e
    pytest.fail("task was not cancelled")


async def _value(error: BaseException) -> BaseException:
    return error


async def test_marked_cancel_sets_cancelled() -> None:
    run = _new_run()
    _tracer()._record_error(run, await _cancel_error(ExpectedCancellation("stop")))
    assert run.status == "cancelled"
    assert run.error  # error text is kept


async def test_marked_cancel_subclass_sets_cancelled() -> None:
    class UserInterrupt(ExpectedCancellation):
        pass

    run = _new_run()
    _tracer()._record_error(run, await _cancel_error(UserInterrupt()))
    assert run.status == "cancelled"


async def test_graph_interrupt_sets_interrupted() -> None:
    run = _new_run()
    _tracer()._record_error(run, GraphInterrupt("pause"))
    assert run.status == "interrupted"
    assert run.error


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: _cancel_error(None),
        lambda: _cancel_error("shutdown"),
        lambda: _value(TimeoutError()),
        lambda: _value(ValueError("boom")),
    ],
)
async def test_unmarked_errors_have_no_status(
    make_error: Callable[[], Awaitable[BaseException]],
) -> None:
    run = _new_run()
    _tracer()._record_error(run, await make_error())
    assert run.status is None
    assert run.error


async def test_status_follows_the_last_error() -> None:
    tracer = _tracer()
    run = _new_run()
    cancel = await _cancel_error(ExpectedCancellation())
    tracer._record_error(run, cancel)
    tracer._record_error(run, ValueError("real error"))
    assert run.status is None
    tracer._record_error(run, cancel)
    assert run.status == "cancelled"


async def test_langsmith_without_status_support_keeps_error_only() -> None:
    run = _OldRun()
    _tracer()._record_error(run, await _cancel_error(ExpectedCancellation()))  # type: ignore[arg-type]
    assert run.error
    assert not hasattr(run, "status")
