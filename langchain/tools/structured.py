from __future__ import annotations

from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    Extra,
    Field,
    create_model,
    validate_arguments,
    validator,
)
from pydantic.generics import GenericModel
from pydantic.main import ModelMetaclass
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager

from langchain.tools.base import BaseTool
from langchain.utilities.async_utils import async_or_sync_call


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


INPUT_T = TypeVar("INPUT_T")
SCHEMA_T = TypeVar("SCHEMA_T", bound=Union[str, BaseModel])
OUTPUT_T = TypeVar("OUTPUT_T")


class BaseStructuredTool(
    GenericModel,
    Generic[INPUT_T, SCHEMA_T, OUTPUT_T],
    BaseModel,
):
    """Parent class for all structured tools."""

    name: str
    description: str
    return_direct: bool = False
    verbose: bool = False
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)
    args_schema: Type[SCHEMA_T]  # :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def args(self) -> dict:
        if isinstance(self.args_schema, BaseModel):
            return self.args_schema.schema()["properties"]
        else:
            return {"tool_input": "str"}

    def _parse_input(self, tool_input: Union[INPUT_T, Any]) -> SCHEMA_T:
        """Load the tool's input into a pydantic model."""
        if not issubclass(self.args_schema, BaseModel):
            raise ValueError(
                f"Tool with args_schema of type {self.args_schema} must overwrite _parse_input."
            )
        # Ignore type because mypy doesn't connect the subclass to the generic SCHEMA_T
        return self.args_schema.parse_obj(tool_input)  # type: ignore

    def _get_verbosity(
        self,
        verbose: Optional[bool] = None,
    ) -> bool:
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        return verbose_

    @abstractmethod
    def _run(self, input_: SCHEMA_T) -> OUTPUT_T:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, input_: SCHEMA_T) -> OUTPUT_T:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: INPUT_T,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> OUTPUT_T:
        """Run the tool."""
        parsed_input = self._parse_input(tool_input)
        verbose_ = self._get_verbosity(verbose)
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            str(tool_input),
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            observation = self._run(parsed_input)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        self.callback_manager.on_tool_end(
            str(observation), verbose=verbose_, color=color, name=self.name, **kwargs
        )
        return observation

    async def arun(
        self,
        tool_input: INPUT_T,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> OUTPUT_T:
        """Run the tool asynchronously."""
        parsed_input = self._parse_input(tool_input)
        verbose_ = self._get_verbosity(verbose)
        await async_or_sync_call(
            self.callback_manager.on_tool_start,
            {"name": self.name, "description": self.description},
            str(parsed_input),
            verbose=verbose_,
            color=start_color,
            is_async=self.callback_manager.is_async,
            **kwargs,
        )
        try:
            # We then call the tool on the tool input to get an observation
            observation = await self._arun(parsed_input)
        except (Exception, KeyboardInterrupt) as e:
            await async_or_sync_call(
                self.callback_manager.on_tool_error,
                e,
                verbose=verbose_,
                is_async=self.callback_manager.is_async,
            )
            raise e
        await async_or_sync_call(
            self.callback_manager.on_tool_end,
            str(observation),
            verbose=verbose_,
            color=color,
            is_async=self.callback_manager.is_async,
            **kwargs,
        )
        return observation

    def __call__(self, tool_input: INPUT_T) -> OUTPUT_T:
        """Make tool callable."""
        return self.run(tool_input)


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


class StructuredTool(BaseStructuredTool[Dict, BaseModel, Any]):
    """StructuredTool that takes in function or coroutine directly."""

    func: Callable[..., Any]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""
    args_schema: Type[BaseModel]  # :meta private:

    @validator("func", pre=True, always=True)
    def validate_func_not_partial(cls, func: Callable) -> Callable:
        """Check that the function is not a partial."""
        if isinstance(func, partial):
            raise ValueError("Partial functions not yet supported in structured tools.")
        return func

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            inferred_model = validate_arguments(self.func).model  # type: ignore
            return get_filtered_args(inferred_model, self.func)

    def _run(self, tool_input: BaseModel) -> Any:
        """Use the tool."""
        return self.func(**tool_input.dict())

    async def _arun(self, tool_input: BaseModel) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(**tool_input.dict())
        raise NotImplementedError(f"StructuredTool {self.name} does not support async")

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "StructuredTool":
        """Make tools out of functions, can be used with or without arguments.

        Args:
            func: The function to run when the tool is called.
            coroutine: The asynchronous version of the function.
            return_direct: Whether to return directly from the tool rather
                than continuing the agent loop.
            args_schema: optional argument schema for user to specify
            infer_schema: Whether to infer the schema of the arguments from
                the function's signature. This also makes the resultant tool
                accept a dictionary input to its `run()` function.
            name: The name of the tool. Defaults to the function name.
            description: The description of the tool. Defaults to the function
                docstring.
        """
        description = func.__doc__ or description
        if description is None or not description.strip():
            raise ValueError(
                f"Function {func.__name__} must have a docstring, or set description."
            )
        name = name or func.__name__
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", func)
        description = f"{name}{signature(func)} - {description}"
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            return_direct=return_direct,
            args_schema=_args_schema,
            description=description,
        )


def structured_tool(
    *args: Union[str, Callable],
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
        return_direct: Whether to return directly from the tool rather
            than continuing the agent loop.
        args_schema: optional argument schema for user to specify
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.

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
        def _make_tool(func: Callable) -> StructuredTool:
            return StructuredTool.from_function(
                name=tool_name,
                func=func,
                args_schema=args_schema,
                return_direct=return_direct,
                infer_schema=infer_schema,
            )

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
        def _partial(func: Callable[[str], str]) -> BaseTool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
