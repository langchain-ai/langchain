import asyncio
import inspect
from functools import wraps
from typing import Any, Callable


def curry(func: Callable[..., Any], **curried_kwargs: Any) -> Callable[..., Any]:
    """Util that wraps a function and partially applies kwargs to it.
    Returns a new function whose signature omits the curried variables.

    Args:
        func: The function to curry.
        curried_kwargs: Arguments to apply to the function.

    Returns:
        A new function with curried arguments applied.

    .. versionadded:: 0.2.14
    """

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwargs = {**curried_kwargs, **kwargs}
        return await func(*args, **new_kwargs)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwargs = {**curried_kwargs, **kwargs}
        return func(*args, **new_kwargs)

    sig = inspect.signature(func)
    # Create a new signature without the curried parameters
    new_params = [p for name, p in sig.parameters.items() if name not in curried_kwargs]

    if asyncio.iscoroutinefunction(func):
        async_wrapper = wraps(func)(async_wrapper)
        setattr(async_wrapper, "__signature__", sig.replace(parameters=new_params))
        return async_wrapper
    else:
        sync_wrapper = wraps(func)(sync_wrapper)
        setattr(sync_wrapper, "__signature__", sig.replace(parameters=new_params))
        return sync_wrapper
