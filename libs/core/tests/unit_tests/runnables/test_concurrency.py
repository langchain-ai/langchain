"""Test concurrency behavior of batch and async batch operations."""

import asyncio
import time
from threading import Lock
from typing import Any

import pytest

from langchain_core.runnables import RunnableConfig, RunnableLambda


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_tasks_on_exception() -> None:
    """Pending tasks are cancelled when one task raises an exception."""
    side_effects: list[str] = []
    never_proceed = asyncio.Event()

    async def worker(x: str) -> str:
        if x == "boom":
            msg = "boom"
            raise ValueError(msg)
        try:
            await never_proceed.wait()
        except asyncio.CancelledError:
            side_effects.append(f"cancelled-{x}")
            raise
        else:
            side_effects.append(f"done-{x}")
            return x

    runnable = RunnableLambda(worker)
    with pytest.raises(ValueError, match="boom"):
        async for _ in runnable.abatch_as_completed(
            ["boom", "a", "b"], config={"max_concurrency": 3}
        ):
            pass

    assert "done-a" not in side_effects
    assert "done-b" not in side_effects
    assert "cancelled-a" in side_effects or "cancelled-b" in side_effects


@pytest.mark.asyncio
async def test_abatch_as_completed_cancels_tasks_on_early_break() -> None:
    """Pending tasks are cancelled when the consumer breaks early."""
    side_effects: list[str] = []

    async def worker(x: str) -> str:
        if x == "fast":
            return "fast"
        await asyncio.sleep(0.2)
        side_effects.append(f"done-{x}")
        return x

    runnable = RunnableLambda(worker)
    async for _idx, _out in runnable.abatch_as_completed(
        ["fast", "slow1", "slow2"], config={"max_concurrency": 3}
    ):
        break

    await asyncio.sleep(1.0)
    assert "done-slow1" not in side_effects
    assert "done-slow2" not in side_effects


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
