from __future__ import annotations

import os
from types import TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

from anyio import CapacityLimiter, Event, create_task_group

T = TypeVar("T")
Results = Dict[Event, Any]


async def execute(
    fn: Callable[..., Awaitable[T]],
    args: Any,
    kwargs: Any,
    limiter: CapacityLimiter,
    results: Results,
    done: Event,
) -> None:
    async with limiter:
        results[done] = await fn(*args, **kwargs)
        done.set()


class AsyncExecutorAwaitable(Generic[T]):
    def __init__(self, done: Event, results: Results) -> None:
        self.done = done
        self.results = results

    def __await__(self) -> Generator[Any, Any, T]:
        # wait for completion
        yield from self.done.wait().__await__()
        # return the result
        return self.results.pop(self.done)


class AsyncExecutor:
    """This is an asyncio equivalent of concurrent.futures.Executor."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.task_group = create_task_group()
        self.limiter = CapacityLimiter(
            max_workers or min(32, (os.cpu_count() or 1) + 4)
        )
        self.results = cast(Results, {})

    def submit(
        self, fn: Callable[..., Awaitable[T]], /, *args: Any, **kwargs: Any
    ) -> AsyncExecutorAwaitable[T]:
        """Submits a callable to be executed with the given arguments.

        Schedules the async callable to be executed as fn(*args, **kwargs), returning
        an awaitable that will return the result of the call.

        Returns:
            An awaitable that will return the result of the call.
        """
        done = Event()

        # start the task
        self.task_group.start_soon(
            execute, fn, args, kwargs, self.limiter, self.results, done
        )
        # return an awaitable for result
        return AsyncExecutorAwaitable[T](done, self.results)

    async def map(self, fn: Callable[..., Awaitable[T]], *iterables: Any) -> List[T]:
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.

        Returns:
            An iterator of results of calling fn on *iterables.

        Raises:
            Exception: If fn(*args) raises for any values.
        """

        # submit all tasks
        tasks = [self.submit(fn, *args) for args in zip(*iterables)]
        # return a list of results
        return [await task for task in tasks]

    async def __aenter__(self) -> AsyncExecutor:
        await self.task_group.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Union[bool, None]:
        return await self.task_group.__aexit__(exc_type, exc_val, exc_tb)
