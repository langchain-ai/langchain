"""Base implementation for tools or skills."""

import asyncio
from abc import abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator, Extra
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


class BaseTool(BaseModel):
    """Class responsible for defining a tool or skill for an LLM."""

    name: str
    description: str
    return_direct: bool = False
    verbose: bool = False
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

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
