from __future__ import annotations

from abc import abstractmethod
from functools import partial
from inspect import signature
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    create_model,
    validate_arguments,
    validator,
)
from pydantic.generics import GenericModel
from pydantic.main import ModelMetaclass

from langchain.tools.base import BaseTool, ToolMixin
from langchain.utilities.async_utils import async_or_sync_call


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
                    f"StructuredTool definition for {name} must include valid type annotations"
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


class BaseStructuredTool(ToolMixin[Dict, Dict], metaclass=ToolMetaclass):
    """Parent class for all structured tools."""

    args_schema: Type[BaseModel]  # :meta private:

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            inferred_model = validate_arguments(self._run).model  # type: ignore
            return get_filtered_args(inferred_model, self._run)

    def _load_parsed_input(self, tool_input: Union[Dict, Any]) -> BaseModel:
        """Load the tool's input into a pydantic model."""
        if not isinstance(tool_input, dict):
            # Despite being typed as a Dict, there are cases when the LLM
            # will not actually output dict args (e.g., for single arg inputs).
            single_field = next(iter(self.args_schema.__fields__))
            tool_input = {single_field: tool_input}
        return self.args_schema.parse_obj(tool_input)

    @abstractmethod
    def _run(self, input_: BaseModel) -> Any:
        """Use the tool."""

    @abstractmethod
    async def _arun(self, input_: BaseModel) -> Any:
        """Use the tool asynchronously."""

    def run(
        self,
        tool_input: Dict,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        parsed_input = self.args_schema.parse_obj(tool_input)
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
        if isinstance(observation, BaseModel):
            observation = observation.dict()
        return str(observation)

    async def arun(
        self,
        tool_input: Union[Dict, Any],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously."""
        parsed_input = self.args_schema.parse_obj(tool_input)
        verbose_ = self._get_verbosity(verbose)
        await async_or_sync_call(
            self.callback_manager.on_tool_start,
            {"name": self.name, "description": self.description},
            str(parsed_input.dict()),
            verbose=verbose_,
            color=start_color,
            is_async=self.callback_manager.is_async,
            **kwargs,
        )
        try:
            # We then call the tool on the tool input to get an observation
            observation = await self._run(parsed_input)
        except (Exception, KeyboardInterrupt) as e:
            await async_or_sync_call(
                self.callback_manager.on_tool_error,
                e,
                verbose=verbose_,
                is_async=self.callback_manager.is_async,
            )
            raise e
        if isinstance(observation, BaseModel):
            observation = observation.dict()
        await async_or_sync_call(
            self.callback_manager.on_tool_end,
            str(observation),
            verbose=verbose_,
            color=color,
            is_async=self.callback_manager.is_async,
            **kwargs,
        )
        return observation

    def __call__(self, tool_input: Dict) -> Any:
        """Make tool callable."""
        return self.run(tool_input)


class StructuredTool(BaseStructuredTool):
    """StructuredTool that takes in function or coroutine directly."""

    description: str = ""
    func: Callable[..., Any]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""

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
        raise NotImplementedError("StructuredTool does not support async")

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


TOOL_T = TypeVar("TOOL_T", bound=ToolMixin, default=BaseTool)
