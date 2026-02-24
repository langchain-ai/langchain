"""Test concurrency behavior of batch and async batch operations."""

import asyncio
import time
from threading import Lock
from typing import Any

import pytest

from langchain_core.runnables import RunnableConfig, RunnableLambda


@pytest.mark.asyncio
async def test_abatch_concurrency() -> None:
    """Test that abatch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        async with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        await asyncio.sleep(0.1)  # Simulate work

        async with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = await runnable.abatch(list(range(num_tasks)), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


@pytest.mark.asyncio
async def test_abatch_as_completed_concurrency() -> None:
    """Test that abatch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        async with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        await asyncio.sleep(0.1)  # Simulate work

        async with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = []
    async for _idx, result in runnable.abatch_as_completed(
        list(range(num_tasks)), config=config
    ):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


def test_batch_concurrency() -> None:
    """Test that batch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0

    lock = Lock()

    def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        time.sleep(0.1)  # Simulate work

        with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = runnable.batch(list(range(num_tasks)), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_pending_on_exception() -> None:
    """Pending tasks are cancelled when the consumer raises an exception."""
    # Use an Event so we wait (with a 1s timeout) for all 4 slow tasks to be
    # cancelled, rather than relying on a fixed number of asyncio.sleep(0) calls.
    cancel_event = asyncio.Event()
    cancel_count = 0

    async def tracked_function(x: Any) -> str:
        nonlocal cancel_count
        # input 0 completes immediately so the consumer body runs; others block.
        if x == 0:
            return "fast"
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancel_count += 1
            if cancel_count == 4:
                cancel_event.set()
            raise
        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)

    async def consume_and_raise() -> None:
        msg = "early exit"
        async for _idx, _result in runnable.abatch_as_completed([0, 1, 2, 3, 4]):
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="early exit"):
        await consume_and_raise()

    # asyncio finalises async generators asynchronously (via GC hooks), so we
    # wait for the event rather than relying on a fixed sleep.
    await asyncio.wait_for(cancel_event.wait(), timeout=1.0)
    assert cancel_count == 4


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_pending_on_break() -> None:
    """Pending tasks are cancelled when the consumer breaks out of the loop."""
    cancel_event = asyncio.Event()
    cancel_count = 0

    async def tracked_function(x: Any) -> str:
        nonlocal cancel_count
        # input 0 completes immediately so we have something to break on.
        if x == 0:
            return "fast"
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancel_count += 1
            if cancel_count == 4:
                cancel_event.set()
            raise
        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)

    results = []
    async for _idx, result in runnable.abatch_as_completed([0, 1, 2, 3, 4]):
        results.append(result)
        break  # stop after first result

    # Wait for all 4 slow tasks to be cancelled.
    await asyncio.wait_for(cancel_event.wait(), timeout=1.0)
    assert results == ["fast"]
    assert cancel_count == 4


def test_batch_as_completed_concurrency() -> None:
    """Test that batch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0

    lock = Lock()

    def tracked_function(x: Any) -> str:
        nonlocal running_tasks, max_running_tasks
        with lock:
            running_tasks += 1
            max_running_tasks = max(max_running_tasks, running_tasks)

        time.sleep(0.1)  # Simulate work

        with lock:
            running_tasks -= 1

        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)
    num_tasks = 10
    max_concurrency = 3

    config = RunnableConfig(max_concurrency=max_concurrency)
    results = []
    for _idx, result in runnable.batch_as_completed(
        list(range(num_tasks)), config=config
    ):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency
