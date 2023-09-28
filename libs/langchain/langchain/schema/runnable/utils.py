from __future__ import annotations

import asyncio
from inspect import signature
from typing import Any, Callable, Coroutine, TypeVar, Union

Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")


async def gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    async with semaphore:
        return await coro


async def gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:
    try:
        return signature(callable).parameters.get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:
    try:
        return signature(callable).parameters.get("config") is not None
    except ValueError:
        return False
