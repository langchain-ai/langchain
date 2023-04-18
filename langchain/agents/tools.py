"""Interface for tools."""
import inspect
from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, Field, create_model

from langchain.tools.base import (
    BaseTool,
)


class Tool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Callable[..., str]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[str]]] = None
    """The asynchronous version of the function."""

    def _run(self, tool_input: Union[str, BaseModel]) -> str:
        """Use the tool."""
        if isinstance(tool_input, str):
            return self.func(tool_input)
        else:
            args, kwargs = _to_args_and_kwargs(tool_input)
            return self.func(*args, **kwargs)

    async def _arun(self, tool_input: Union[str, BaseModel]) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            if isinstance(tool_input, str):
                return await self.coroutine(tool_input)
            else:
                args, kwargs = _to_args_and_kwargs(tool_input)
                return await self.coroutine(*args, **kwargs)
        raise NotImplementedError("Tool does not support async")

    # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self, name: str, func: Callable[[str], str], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super(Tool, self).__init__(
            name=name, func=func, description=description, **kwargs
        )


class InvalidTool(BaseTool):
    """Tool that is run when invalid tool name is encountered by agent."""

    name = "invalid_tool"
    description = "Called when tool name is invalid."

    def _run(self, tool_name: Union[str, BaseModel]) -> str:
        """Use the tool."""
        return f"{str(tool_name)} is not a valid tool, try another one."

    async def _arun(self, tool_name: Union[str, BaseModel]) -> str:
        """Use the tool asynchronously."""
        return f"{str(tool_name)} is not a valid tool, try another one."


def _to_args_and_kwargs(model: BaseModel) -> Tuple[Sequence, dict]:
    """Convert pydantic model to args and kwargs."""
    args = []
    kwargs = {}
    for name, field in model.__fields__.items():
        value = getattr(model, name)
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


def _create_args_schema_model_from_signature(run_func: Callable) -> Type[BaseModel]:
    """Create a pydantic model type from a function's signature."""
    signature_ = inspect.signature(run_func)
    field_definitions: Dict[str, Any] = {}

    for name, param in signature_.parameters.items():
        if name == "self":
            continue
        default_value = (
            param.default if param.default != inspect.Parameter.empty else None
        )
        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )
        # Handle functions with *args in the signature
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            field_definitions[name] = (
                Any,
                Field(default=None, extra={"is_var_positional": True}),
            )
        # handle functions with **kwargs in the signature
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            field_definitions[name] = (
                Any,
                Field(default=None, extra={"is_var_keyword": True}),
            )
        # Handle all other named parameters
        else:
            is_keyword_only = param.kind == inspect.Parameter.KEYWORD_ONLY
            field_definitions[name] = (
                annotation,
                Field(
                    default=default_value, extra={"is_keyword_only": is_keyword_only}
                ),
            )
    return create_model("ArgsModel", **field_definitions)  # type: ignore


def _create_schema_if_multiarg(
    func: Callable,
) -> Optional[Type[BaseModel]]:
    signature_ = inspect.signature(func)
    parameters = signature_.parameters
    if len(parameters) == 1 and next(iter(parameters.values())).annotation == str:
        # Default tools take a single string as input and don't need a dynamic
        # schema validation
        return None
    else:
        return _create_args_schema_model_from_signature(func)


def tool(*args: Union[str, Callable], return_direct: bool = False) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

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
        def _make_tool(func: Callable) -> Tool:
            assert func.__doc__, "Function must have a docstring"
            # Description example:
            # search_api(query: str) - Searches the API for the query.
            description = f"{tool_name}{signature(func)} - {func.__doc__.strip()}"
            args_schema = _create_schema_if_multiarg(func)
            tool_ = Tool(
                name=tool_name,
                func=func,
                args_schema=args_schema,
                description=description,
                return_direct=return_direct,
            )
            return tool_

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
