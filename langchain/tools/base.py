"""Base implementation for tools or skills."""

from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, Extra, Field, validate_arguments, validator

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


def _to_args_and_kwargs(run_input: Union[str, Dict]) -> Tuple[Sequence, dict]:
    # For backwards compatability, if run_input is a string,
    # pass as a positional argument.
    if isinstance(run_input, str):
        return (run_input,), {}
    else:
        return [], run_input


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
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            inferred_model = validate_arguments(self._run).model  # type: ignore
            schema = inferred_model.schema()["properties"]
            valid_keys = signature(self._run).parameters
            return {k: schema[k] for k in valid_keys}

    def _parse_input(
        self,
        tool_input: Union[str, Dict],
    ) -> None:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
        else:
            if input_args is not None:
                input_args.validate(tool_input)

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
        self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            tool_args, tool_kwargs = _to_args_and_kwargs(tool_input)
            observation = self._run(*tool_args, **tool_kwargs)
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
        self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        if self.callback_manager.is_async:
            await self.callback_manager.on_tool_start(
                {"name": self.name, "description": self.description},
                tool_input if isinstance(tool_input, str) else str(tool_input),
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
            args, kwargs = _to_args_and_kwargs(tool_input)
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
