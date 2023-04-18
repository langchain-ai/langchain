"""Base implementation for tools or skills."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, Extra, Field, validator

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


def _to_args_and_kwargs(run_input: Union[str, BaseModel]) -> Tuple[Sequence, dict]:
    # For backwards compatability, if run_input is a string,
    # pass as a positional argument.
    if isinstance(run_input, str):
        return (run_input,), {}
    args = []
    kwargs = {}
    for name, field in run_input.__fields__.items():
        value = getattr(run_input, name)
        # Handle *args in the function signature
        if field.field_info.extra.get("extra", {}).get("is_var_positional"):
            if isinstance(value, str):
                # Base case for backwards compatability
                args.append(value)
            elif value is not None:
                args.extend(value)
        # Handle **kwargs in the function signature
        elif field.field_info.extra.get("extra", {}).get("is_var_keyword"):
            if value is not None:
                kwargs.update(value)
        elif field.field_info.extra.get("extra", {}).get("is_keyword_only"):
            kwargs[name] = value
        else:
            args.append(value)

    return tuple(args), kwargs


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
    def args(self) -> Union[Type[BaseModel], Type[str]]:
        """Generate an input pydantic model."""
        return str if self.args_schema is None else self.args_schema

    def _parse_input(
        self,
        tool_input: Union[str, Dict],
    ) -> Union[BaseModel, str]:
        """Convert tool input to pydantic model."""
        input_args = self.args
        if isinstance(tool_input, str):
            if issubclass(input_args, BaseModel):
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.parse_obj({key_: tool_input})
            # Passing as a positional argument is more straightforward for
            # backwards compatability
            return tool_input
        if issubclass(input_args, BaseModel):
            return input_args.parse_obj(tool_input)
        else:
            raise ValueError(
                f"args_schema required for tool {self.name} in order to"
                f" accept input of type {type(tool_input)}"
            )

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
        **kwargs: Any,
    ) -> str:
        """Run the tool."""
        run_input = self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            run_input if isinstance(run_input, str) else str(run_input.dict()),
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            args, kwargs = _to_args_and_kwargs(run_input)
            observation = self._run(*args, **kwargs)
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
        **kwargs: Any,
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
                run_input if isinstance(run_input, str) else str(run_input.dict()),
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        else:
            self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                run_input if isinstance(run_input, str) else str(run_input.dict()),
                verbose=verbose_,
                color=start_color,
                **kwargs,
            )
        try:
            # We then call the tool on the tool input to get an observation
            args, kwargs = _to_args_and_kwargs(run_input)
            observation = await self._arun(*args, **kwargs)
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
