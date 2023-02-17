"""Base implementation for tools or skills."""

import asyncio
from abc import abstractmethod
from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator, Extra
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import AgentAction


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
    def _run(self, tool_input: str) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Make tools callable by piping through to `func`."""
        if asyncio.iscoroutinefunction(self._run):
            raise TypeError("Coroutine cannot be called directly")
        return self._run(*args, **kwargs)

    def run(self, action: AgentAction, **kwargs) -> str:
        """Run the tool."""
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            action,
            verbose=self.verbose,
        )
        try:
            observation = self._run(action.tool_input)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(
                e, verbose=self.verbose
            )
            raise e
        self.callback_manager.on_tool_end(
            observation,
            verbose=self.verbose,
            **kwargs
        )
        return observation

    async def arun(self, action: AgentAction, **kwargs) -> str:
        """Run the tool asynchronously."""
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                action,
                verbose=self.verbose,
            )
        else:
            self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                action,
                verbose=self.verbose,
            )
        try:
            # We then call the tool on the tool input to get an observation
            observation = await self._arun(action.tool_input)
        except (Exception, KeyboardInterrupt) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_tool_error(e, verbose=self.verbose)
            else:
                self.callback_manager.on_tool_error(e, verbose=self.verbose)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_end(
                observation,
                verbose=self.verbose,
                **kwargs
            )
        else:
            self.callback_manager.on_tool_end(
                observation,
                verbose=self.verbose,
                **kwargs
            )
        return observation


class BaseToolkit(BaseModel):
    """Class responsible for defining a collection of related tools."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
