"""Base implementation for tools or skills."""

import asyncio
from abc import abstractmethod
from typing import Any, List

from pydantic import BaseModel


class BaseTool(BaseModel):
    """Class responsible for defining a tool or skill for an LLM."""

    name: str
    description: str
    return_direct: bool = False

    @abstractmethod
    def func(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""

    @abstractmethod
    async def afunc(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool asynchronously."""

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Make tools callable by piping through to `func`."""
        if asyncio.iscoroutinefunction(self.func):
            raise TypeError("Coroutine cannot be called directly")
        return self.func(*args, **kwargs)


class BaseToolkit(BaseModel):
    """Class responsible for defining a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
