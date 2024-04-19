from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from cassandra.cluster import ResponseFuture


async def wrapped_response_future(
    func: Callable[..., ResponseFuture], *args: Any, **kwargs: Any
) -> Any:
    loop = asyncio.get_event_loop()
    asyncio_future = loop.create_future()
    response_future = func(*args, **kwargs)

    def success_handler(_: Any) -> None:
        loop.call_soon_threadsafe(asyncio_future.set_result, response_future.result())

    def error_handler(exc: BaseException) -> None:
        loop.call_soon_threadsafe(asyncio_future.set_exception, exc)

    response_future.add_callbacks(success_handler, error_handler)
    return await asyncio_future
