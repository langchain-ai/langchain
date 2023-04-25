"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union

from pydantic import (
    BaseModel,
    Extra,
    Field,
    create_model,
    validate_arguments,
    validator,
)
from pydantic.main import ModelMetaclass

from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager


def _to_args_and_kwargs(run_input: Union[str, Dict]) -> Tuple[Sequence, dict]:
    # For backwards compatability, if run_input is a string,
    # pass as a positional argument.
    if isinstance(run_input, str):
        return (run_input,), {}
    else:
        return [], run_input


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


class ToolMetaclass(ModelMetaclass):
    """Metaclass for BaseTool to ensure the provided args_schema

    doesn't silently ignored."""

    def __new__(
        cls: Type[ToolMetaclass], name: str, bases: Tuple[Type, ...], dct: dict
    ) -> ToolMetaclass:
        """Create the definition of the new tool class."""
        schema_type: Optional[Type[BaseModel]] = dct.get("args_schema")
        if schema_type is not None:
            schema_annotations = dct.get("__annotations__", {})
            args_schema_type = schema_annotations.get("args_schema", None)
            if args_schema_type is None or args_schema_type == BaseModel:
                # Throw errors for common mis-annotations.
                # TODO: Use get_args / get_origin and fully
                # specify valid annotations.
                typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
                raise SchemaAnnotationError(
                    f"Tool definition for {name} must include valid type annotations"
                    f" for argument 'args_schema' to behave as expected.\n"
                    f"Expected annotation of 'Type[BaseModel]'"
                    f" but got '{args_schema_type}'.\n"
                    f"Expected class looks like:\n"
                    f"{typehint_mandate}"
                )
        # Pass through to Pydantic's metaclass
        return super().__new__(cls, name, bases, dct)


def _create_subset_model(
    name: str, model: BaseModel, field_names: list
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {
        field_name: (
            model.__fields__[field_name].type_,
            model.__fields__[field_name].default,
        )
        for field_name in field_names
        if field_name in model.__fields__
    }
    return create_model(name, **fields)  # type: ignore


def get_filtered_args(inferred_model: Type[BaseModel], func: Callable) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.schema()["properties"]
    valid_keys = signature(func).parameters
    return {k: schema[k] for k in valid_keys}


def create_schema_from_function(model_name: str, func: Callable) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature."""
    inferred_model = validate_arguments(func).model  # type: ignore
    # Pydantic adds placeholder virtual fields we need to strip
    filtered_args = get_filtered_args(inferred_model, func)
    return _create_subset_model(
        f"{model_name}Schema", inferred_model, list(filtered_args)
    )


class BaseTool(ABC, BaseModel, metaclass=ToolMetaclass):
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
            return get_filtered_args(inferred_model, self._run)

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
