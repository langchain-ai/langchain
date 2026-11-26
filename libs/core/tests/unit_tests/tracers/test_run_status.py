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


class GraphInterrupt(Exception):  # noqa: N818
    """Interrupt tagged with an explicit run status, as LangGraph does."""

    ls_run_status = "interrupted"


class GraphDrained(Exception):  # noqa: N818
    """Shutdown drain: carries no `ls_run_status`."""


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


def test_ls_run_status_attribute_cancelled_is_honored() -> None:
    class Stopped(Exception):  # noqa: N818
        ls_run_status = "cancelled"

    run = _new_run()
    _tracer()._record_error(run, Stopped())
    assert run.status == "cancelled"


@pytest.mark.parametrize("value", ["success", "error", "pending", "", None, 1])
def test_other_ls_run_status_values_are_ignored(value: object) -> None:
    error = type("Weird", (Exception,), {"ls_run_status": value})()
    run = _new_run()
    _tracer()._record_error(run, error)
    assert run.status is None
    assert run.error


def test_graph_drained_has_no_status() -> None:
    run = _new_run()
    _tracer()._record_error(run, GraphDrained("shutdown"))
    assert run.status is None
    assert run.error


def test_class_named_graph_interrupt_without_attribute_is_an_error() -> None:
    error = type("GraphInterrupt", (Exception,), {})("pause")
    run = _new_run()
    _tracer()._record_error(run, error)
    assert run.status is None
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


def test_bare_cancel_during_marked_cancel_cleanup() -> None:
    marked = asyncio.CancelledError(ExpectedCancellation("user"))
    bare = asyncio.CancelledError()
    bare.__context__ = marked
    run = _new_run()
    _tracer()._record_error(run, bare)
    # The message survives task boundaries, so the bare cancel is a separate
    # (unmarked) cancel.
    assert run.status is None


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
