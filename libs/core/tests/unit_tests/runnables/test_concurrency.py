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


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_pending_tasks_on_exception() -> None:
    """Test that pending tasks are cancelled when one task raises."""
    blocked_task_inputs: list[str] = []
    cancelled_task_inputs: list[str] = []
    block = asyncio.Event()

    async def tracked_function(x: str) -> str:
        if x == "bad":
            await asyncio.sleep(0.01)
            msg = "boom"
            raise RuntimeError(msg)

        blocked_task_inputs.append(x)
        try:
            await block.wait()
        except asyncio.CancelledError:
            cancelled_task_inputs.append(x)
            raise

        return f"Completed {x}"

    runnable = RunnableLambda(tracked_function)

    with pytest.raises(RuntimeError, match="boom"):
        async for _idx, _result in runnable.abatch_as_completed(["bad", "a", "b"]):
            pass

    await asyncio.sleep(0)
    assert set(blocked_task_inputs) == {"a", "b"}
    assert set(cancelled_task_inputs) == {"a", "b"}


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_pending_tasks_on_early_exit() -> None:
    """Test that pending tasks are cancelled when iteration exits early."""
    completed_task_inputs: list[int] = []

    async def tracked_function(x: int) -> int:
        if x == 0:
            await asyncio.sleep(0.01)
            return x

        await asyncio.sleep(0.2)
        completed_task_inputs.append(x)
        return x

    runnable = RunnableLambda(tracked_function)

    async for _idx, result in runnable.abatch_as_completed([0, 1, 2]):
        assert result == 0
        break

    await asyncio.sleep(0.25)
    assert completed_task_inputs == []


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
