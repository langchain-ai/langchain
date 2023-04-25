"""Async utilities."""
from typing import Any, Callable


async def async_or_sync_call(
    method: Callable, *args: Any, is_async: bool, **kwargs: Any
) -> Any:
    """Run the callback manager method asynchronously or synchronously."""
    if is_async:
        return await method(*args, **kwargs)
    else:
        return method(*args, **kwargs)
