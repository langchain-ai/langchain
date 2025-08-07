import inspect
from collections.abc import Awaitable
from typing import Any, Callable

from langchain_core.tools import tool


def _make_wrapped_func(func: Callable[..., str]) -> Callable[..., list[dict[str, Any]]]:
    def wrapped(x: str) -> list[dict[str, Any]]:
        return [{"type": "custom_tool_call_output", "output": func(x)}]

    return wrapped


def _make_wrapped_coroutine(
    coroutine: Callable[..., Awaitable[str]],
) -> Callable[..., Awaitable[list[dict[str, Any]]]]:
    async def wrapped(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = await coroutine(*args, **kwargs)
        return [{"type": "custom_tool_call_output", "output": result}]

    return wrapped


def custom_tool(*args: Any, **kwargs: Any) -> Any:
    def decorator(func: Callable[..., Any]) -> Any:
        metadata = {"type": "custom_tool"}
        if "format" in kwargs:
            metadata["format"] = kwargs.pop("format")
        tool_obj = tool(infer_schema=False, **kwargs)(func)
        tool_obj.metadata = metadata
        tool_obj.description = func.__doc__
        if inspect.iscoroutinefunction(func):
            tool_obj.coroutine = _make_wrapped_coroutine(func)
        else:
            tool_obj.func = _make_wrapped_func(func)
        return tool_obj

    if args and callable(args[0]) and not kwargs:
        return decorator(args[0])

    return decorator
