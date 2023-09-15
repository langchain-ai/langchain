import asyncio

import pytest

from langchain.utils.aexecutor import AsyncExecutor


@pytest.mark.asyncio
async def test_aexecutor_one_task() -> None:
    """Test that we can run a single task."""

    async def return_one() -> int:
        return 1

    async with AsyncExecutor() as executor:
        assert await executor.submit(return_one) == 1

    with pytest.raises(RuntimeError):
        executor.submit(return_one)


@pytest.mark.asyncio
async def test_aexecutor_several_tasks_sequence() -> None:
    """Test that we can run several tasks in sequence."""

    async def identity(arg: int) -> int:
        return arg

    async with AsyncExecutor() as executor:
        results = []
        results.append(await executor.submit(identity, 1))
        results.append(await executor.submit(identity, 2))
        results.append(await executor.submit(identity, 3))
        results.append(await executor.submit(identity, 4))
        results.append(await executor.submit(identity, 5))
        assert results == [1, 2, 3, 4, 5]

    with pytest.raises(RuntimeError):
        executor.submit(identity, 1)


@pytest.mark.asyncio
@pytest.mark.timeout(2)
async def test_aexecutor_several_tasks_concurrent() -> None:
    """Test that we can run submit several tasks, and then await them afterwards."""

    async def identity(arg: int) -> int:
        await asyncio.sleep(1)
        return arg

    async with AsyncExecutor() as executor:
        tasks = [executor.submit(identity, i) for i in range(1, 6)]
        assert [await task for task in tasks] == [1, 2, 3, 4, 5]

    with pytest.raises(RuntimeError):
        executor.submit(identity, 1)


@pytest.mark.asyncio
@pytest.mark.timeout(2)
async def test_aexecutor_several_tasks_exception() -> None:
    """Test that we can run submit several tasks, and then await them afterwards."""

    async def identity(arg: int) -> int:
        if arg == 3:
            raise ValueError("3 is not allowed")
        await asyncio.sleep(1)
        return arg

    with pytest.raises(Exception):  # ExceptionGroup in >=3.11
        async with AsyncExecutor() as executor:
            tasks = [executor.submit(identity, i) for i in range(1, 6)]
            assert [await task for task in tasks] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
@pytest.mark.timeout(2)
async def test_aexecutor_map() -> None:
    """Test that we can run submit several tasks, and then await them afterwards."""

    async def identity(arg: int) -> int:
        await asyncio.sleep(1)
        return arg

    async with AsyncExecutor() as executor:
        assert await executor.map(identity, range(1, 6)) == [1, 2, 3, 4, 5]

    with pytest.raises(RuntimeError):
        executor.submit(identity, 1)
