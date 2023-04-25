"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    Extra,
    Field,
    validator,
)

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


T = TypeVar("T")


class ToolMixin(BaseModel):
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

    def _get_verbosity(
        self,
        verbose: Optional[bool] = None,
    ) -> bool:
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        return verbose_

    def _get_input(self, tool_input: T) -> T:
        return tool_input


class BaseTool(ABC, ToolMixin):
    """Interface LangChain tools must implement."""

    @abstractmethod
    def _run(self, tool_input: str) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: str,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Run the tool."""
        verbose_ = self._get_verbosity(verbose)
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input,
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            observation = self._run(tool_input)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        self.callback_manager.on_tool_end(
            observation, verbose=verbose_, color=color, name=self.name, **kwargs
        )
        return observation

    async def arun(
        self,
        tool_input: str,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> str:
        """Run the tool asynchronously."""
        verbose_ = self._get_verbosity(verbose)
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                tool_input,
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        else:
            self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                tool_input if isinstance(tool_input, str) else str(tool_input),
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        try:
            # We then call the tool on the tool input to get an observation
            observation = await self._arun(tool_input)
        except (Exception, KeyboardInterrupt) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_tool_error(e, verbose=verbose_)
            else:
                self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_end(
                observation, verbose=verbose_, color=color, name=self.name, **kwargs
            )
        else:
            self.callback_manager.on_tool_end(
                observation, verbose=verbose_, color=color, name=self.name, **kwargs
            )
        return observation

    def __call__(self, tool_input: str) -> str:
        """Make tool callable."""
        return self.run(tool_input)
