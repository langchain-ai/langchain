"""Regression tests for chain-group tracing finalization on BaseException.

`trace_as_chain_group` / `atrace_as_chain_group` used to catch only `Exception`,
so a `BaseException` such as `asyncio.CancelledError` skipped both `on_chain_error`
and `on_chain_end`, leaving the group's run pending.

See https://github.com/langchain-ai/langchain/issues/39163.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from langchain_core.callbacks.manager import (
    atrace_as_chain_group,
    trace_as_chain_group,
)
from tests.unit_tests.fake.callbacks import (
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def test_trace_as_chain_group_finalizes_run_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = FakeCallbackHandler()
    # The context managers only attach the supplied callbacks when tracing is
    # enabled; short-circuit lookup so a run is created without a LangChainTracer.
    monkeypatch.setattr(
        "langchain_core.tracers.context._get_trace_callbacks",
        MagicMock(return_value=[handler]),
    )

    with pytest.raises(KeyboardInterrupt), trace_as_chain_group("test_group"):
        raise KeyboardInterrupt

    assert handler.errors == 1
    assert handler.chain_ends == 0


async def test_atrace_as_chain_group_finalizes_run_on_task_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = FakeAsyncCallbackHandler()
    monkeypatch.setattr(
        "langchain_core.tracers.context._get_trace_callbacks",
        MagicMock(return_value=[handler]),
    )

    entered = asyncio.Event()

    async def _run() -> None:
        async with atrace_as_chain_group("test_group"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(_run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # `@shielded` `on_chain_error` may still be draining after the cancelled task
    # returns; yield so the recording handler can finish.
    await asyncio.sleep(0)
    assert handler.errors == 1
    assert handler.chain_ends == 0
