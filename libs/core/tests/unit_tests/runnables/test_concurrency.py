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


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_on_exception() -> None:
    """Test that abatch_as_completed cancels pending tasks when one raises."""
    completed_count = 0

    async def slow_function(x: Any) -> str:
        nonlocal completed_count
        if x == 0:
            msg = "task failed"
            raise ValueError(msg)
        # Remaining tasks sleep long enough to still be pending when the
        # exception propagates and the generator is closed.
        await asyncio.sleep(5)
        completed_count += 1
        return f"Completed {x}"

    runnable = RunnableLambda(slow_function)

    with pytest.raises(ValueError, match="task failed"):
        async for _idx, _result in runnable.abatch_as_completed(
            list(range(5)),
        ):
            pass

    # Give the event loop a chance to process cancellations
    await asyncio.sleep(0.1)

    # No long-running tasks should have completed because they were cancelled
    assert completed_count == 0


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_on_early_break() -> None:
    """Test that abatch_as_completed cancels pending tasks on early break."""
    completed_count = 0
    started = asyncio.Event()

    async def slow_function(x: Any) -> str:
        nonlocal completed_count
        if x == 0:
            started.set()
            return "first"
        # Wait for the first task to finish so it can be yielded, then sleep
        # long enough that these tasks are still running when the caller breaks.
        await started.wait()
        await asyncio.sleep(5)
        completed_count += 1
        return f"Completed {x}"

    runnable = RunnableLambda(slow_function)

    async for _idx, _result in runnable.abatch_as_completed(list(range(5))):
        # Break after receiving the first result
        break

    # Give the event loop a chance to process cancellations
    await asyncio.sleep(0.1)

    # No long-running tasks should have completed because they were cancelled
    assert completed_count == 0
