"""Interface for tools."""
import asyncio
from dataclasses import dataclass
from inspect import signature
from typing import Any, Awaitable, Callable, Optional, Union


@dataclass
class Tool:
    """Interface for tools."""

    name: str
    func: Callable[[str], str]
    description: Optional[str] = None
    return_direct: bool = False
    # If the tool has a coroutine, then we can use this to run it asynchronously
    coroutine: Optional[Callable[[str], Awaitable[str]]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Make tools callable by piping through to `func`."""
        if asyncio.iscoroutinefunction(self.func):
            raise TypeError("Coroutine cannot be called directly")
        return self.func(*args, **kwargs)


def tool(
    *args: Union[str, Callable], return_direct: bool = False
) -> Union[Callable, Tool]:
    """Make tools out of functions, can be used with or without arguments.

    Requires:
        - Function must be of type (str) -> str
        - Function must have a docstring

    Examples:
        .. code-block:: python

            @tool
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return

            @tool("search", return_direct=True)
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(func: Callable[[str], str]) -> Tool:
            assert func.__doc__, "Function must have a docstring"
            # Description example:
            #   search_api(query: str) - Searches the API for the query.
            description = f"{tool_name}{signature(func)} - {func.__doc__.strip()}"
            tool = Tool(
                name=tool_name,
                func=func,
                description=description,
                return_direct=return_direct,
            )
            return tool

        return _make_tool

    if len(args) == 1 and isinstance(args[0], str):
        # if the argument is a string, then we use the string as the tool name
        # Example usage: @tool("search", return_direct=True)
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # if the argument is a function, then we use the function name as the tool name
        # Example usage: @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # if there are no arguments, then we use the function name as the tool name
        # Example usage: @tool(return_direct=True)
        def _partial(func: Callable[[str], str]) -> Tool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
