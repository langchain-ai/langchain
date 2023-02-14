"""Base implementation for tools or skills."""

import asyncio
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel


class Tool(BaseModel):
    """Class responsible for defining a tool or skill for an LLM."""

    name: str
    description: str
    return_direct: bool = False
    # If the tool has a coroutine, then we can use this to run it asynchronously
    coroutine: Optional[Callable[[str], Awaitable[str]]] = None

    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Make tools callable by piping through to `func`."""
        if asyncio.iscoroutinefunction(self.func):
            raise TypeError("Coroutine cannot be called directly")
        return self.func(*args, **kwargs)
