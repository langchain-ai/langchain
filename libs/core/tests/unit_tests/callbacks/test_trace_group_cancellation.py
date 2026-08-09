"""Regression tests for chain-group tracing finalization on cancellation.

`trace_as_chain_group` / `atrace_as_chain_group` used to catch only ``Exception``,
so a ``BaseException`` such as ``asyncio.CancelledError`` (raised when an
ASGI/WebSocket client disconnects) slipped past both the ``except`` and the
``else`` branch, and the group's run was never finalized - it stayed pending.

See https://github.com/langchain-ai/langchain/issues/39163.
"""

import asyncio
from typing import Any

import pytest

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.manager import (
    atrace_as_chain_group,
    trace_as_chain_group,
)


class _NonException(BaseException):
    """A BaseException that is not an Exception (like asyncio.CancelledError)."""


def test_trace_as_chain_group_finalizes_run_on_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Recorder(BaseCallbackHandler):
        def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            events.append("error")

        def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
            events.append("end")

    # The context managers only open a run when tracing is enabled; short-circuit
    # the trace-callback lookup so a run is created without a real LangChainTracer.
    monkeypatch.setattr(
        "langchain_core.tracers.context._get_trace_callbacks",
        lambda *args, **kwargs: [Recorder()],
    )

    with pytest.raises(_NonException):
        with trace_as_chain_group("test_group"):
            raise _NonException

    # The run is finalized as an error, not left pending, and not double-ended.
    assert events == ["error"]


async def test_atrace_as_chain_group_finalizes_run_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Recorder(AsyncCallbackHandler):
        async def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            events.append("error")

        async def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
            events.append("end")

    monkeypatch.setattr(
        "langchain_core.tracers.context._get_trace_callbacks",
        lambda *args, **kwargs: [Recorder()],
    )

    with pytest.raises(asyncio.CancelledError):
        async with atrace_as_chain_group("test_group"):
            raise asyncio.CancelledError

    assert events == ["error"]
