"""Test concurrency behavior of batch and async batch operations."""

import asyncio
import time

import pytest

from langchain_core.runnables import RunnableConfig, RunnableLambda


@pytest.mark.asyncio
async def test_abatch_concurrency():
    """Test that abatch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: int) -> str:
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
    results = await runnable.abatch(range(num_tasks), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


@pytest.mark.asyncio
async def test_abatch_as_completed_concurrency():
    """Test that abatch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    lock = asyncio.Lock()

    async def tracked_function(x: int) -> str:
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
    async for idx, result in runnable.abatch_as_completed(
        range(num_tasks), config=config
    ):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency, (
        f"abatch_as_completed exceeded max_concurrency: "
        f"got {max_running_tasks}, expected <= {max_concurrency}"
    )


def test_batch_concurrency():
    """Test that batch respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    from threading import Lock

    lock = Lock()

    def tracked_function(x: int) -> str:
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
    results = runnable.batch(range(num_tasks), config=config)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency


def test_batch_as_completed_concurrency():
    """Test that batch_as_completed respects max_concurrency."""
    running_tasks = 0
    max_running_tasks = 0
    from threading import Lock

    lock = Lock()

    def tracked_function(x: int) -> str:
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
    for idx, result in runnable.batch_as_completed(range(num_tasks), config=config):
        results.append(result)

    assert len(results) == num_tasks
    assert max_running_tasks <= max_concurrency
