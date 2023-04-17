"""Base implementation for tools or skills."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, Extra, Field, create_model, validator

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


class BaseTool(ABC, BaseModel):
    """Interface LangChain tools must implement."""

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""
    return_direct: bool = False
    verbose: bool = False
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def args(self) -> Type[BaseModel]:
        """Generate an input pydantic model."""
        if self.args_schema is not None:
            return self.args_schema

        signature_ = inspect.signature(self._run)
        field_definitions: Dict[str, Tuple[Any, Optional[Any]]] = {}
        for name, param in signature_.parameters.items():
            if name == "self":
                continue
            default_value = (
                param.default if param.default != inspect.Parameter.empty else None
            )
            field_definitions[name] = (param.annotation, default_value)
        return create_model("ArgsModel", **field_definitions)  # type: ignore

    def _parse_input(
        self,
        tool_input: Union[str, Dict],
    ) -> BaseModel:
        """Convert tool input to pydantic model."""
        pydantic_input_type = self.args
        if isinstance(tool_input, str):
            # For backwards compatibility, a tool that only takes
            # a single string input will be converted to a dict.
            # to be validated.
            field_name = next(iter(pydantic_input_type.__fields__))
            tool_input = {field_name: tool_input}
        return pydantic_input_type.parse_obj(tool_input)

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any
    ) -> str:
        """Run the tool."""
        run_input = self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            str(run_input.dict()),
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            observation = self._run(**run_input.dict())
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        self.callback_manager.on_tool_end(
            observation, verbose=verbose_, color=color, name=self.name, **kwargs
        )
        return observation

    async def arun(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any
    ) -> str:
        """Run the tool asynchronously."""
        run_input = self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                str(run_input.dict()),
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        else:
            self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                str(run_input.dict()),
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        try:
            # We then call the tool on the tool input to get an observation
            observation = await self._arun(**run_input.dict())
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
