"""Convert functions and runnables to tools."""

import inspect
from typing import Any, Callable, Literal, Optional, Union, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import Runnable
from langchain_core.tools.base import ArgsSchema, BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.tools.structured import StructuredTool


@overload
def tool(
    *,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> Callable[[Union[Callable, Runnable]], BaseTool]: ...


@overload
def tool(
    name_or_callable: str,
    runnable: Runnable,
    *,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: Callable,
    *,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: str,
    *,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> Callable[[Union[Callable, Runnable]], BaseTool]: ...


def tool(
    name_or_callable: Optional[Union[str, Callable]] = None,
    runnable: Optional[Runnable] = None,
    *args: Any,
    description: Optional[str] = None,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> Union[
    BaseTool,
    Callable[[Union[Callable, Runnable]], BaseTool],
]:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        name_or_callable: Optional name of the tool or the callable to be
            converted to a tool. Must be provided as a positional argument.
        runnable: Optional runnable to convert to a tool. Must be provided as a
            positional argument.
        description: Optional description for the tool.
            Precedence for the tool description value is as follows:
                - `description` argument
                    (used even if docstring and/or `args_schema` are provided)
                - tool function docstring
                    (used even if `args_schema` is provided)
                - `args_schema` description
                    (used only if `description` / docstring are not provided)
        *args: Extra positional arguments. Must be empty.
        return_direct: Whether to return directly from the tool rather
            than continuing the agent loop. Defaults to False.
        args_schema: optional argument schema for user to specify.
            Defaults to None.
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.
            Defaults to True.
        response_format: The tool response format. If "content" then the output of
            the tool is interpreted as the contents of a ToolMessage. If
            "content_and_artifact" then the output is expected to be a two-tuple
            corresponding to the (content, artifact) of a ToolMessage.
            Defaults to "content".
        parse_docstring: if ``infer_schema`` and ``parse_docstring``, will attempt to
            parse parameter descriptions from Google Style function docstrings.
            Defaults to False.
        error_on_invalid_docstring: if ``parse_docstring`` is provided, configure
            whether to raise ValueError on invalid Google Style docstrings.
            Defaults to True.

    Returns:
        The tool.

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

            @tool(response_format="content_and_artifact")
            def search_api(query: str) -> tuple[str, dict]:
                return "partial json of results", {"full": "object of results"}

    .. versionadded:: 0.2.14
    Parse Google-style docstrings:

        .. code-block:: python

            @tool(parse_docstring=True)
            def foo(bar: str, baz: int) -> str:
                \"\"\"The foo.

                Args:
                    bar: The bar.
                    baz: The baz.
                \"\"\"
                return bar

            foo.args_schema.model_json_schema()

        .. code-block:: python

            {
                "title": "foo",
                "description": "The foo.",
                "type": "object",
                "properties": {
                    "bar": {
                        "title": "Bar",
                        "description": "The bar.",
                        "type": "string"
                    },
                    "baz": {
                        "title": "Baz",
                        "description": "The baz.",
                        "type": "integer"
                    }
                },
                "required": [
                    "bar",
                    "baz"
                ]
            }

        Note that parsing by default will raise ``ValueError`` if the docstring
        is considered invalid. A docstring is considered invalid if it contains
        arguments not in the function signature, or is unable to be parsed into
        a summary and "Args:" blocks. Examples below:

        .. code-block:: python

            # No args section
            def invalid_docstring_1(bar: str, baz: int) -> str:
                \"\"\"The foo.\"\"\"
                return bar

            # Improper whitespace between summary and args section
            def invalid_docstring_2(bar: str, baz: int) -> str:
                \"\"\"The foo.
                Args:
                    bar: The bar.
                    baz: The baz.
                \"\"\"
                return bar

            # Documented args absent from function signature
            def invalid_docstring_3(bar: str, baz: int) -> str:
                \"\"\"The foo.

                Args:
                    banana: The bar.
                    monkey: The baz.
                \"\"\"
                return bar
    """  # noqa: D214,D405,D410,D411,D412,D416

    def _create_tool_factory(
        tool_name: str,
    ) -> Callable[[Union[Callable, Runnable]], BaseTool]:
        """Create a decorator that takes a callable and returns a tool.

        Args:
            tool_name: The name that will be assigned to the tool.

        Returns:
            A function that takes a callable or Runnable and returns a tool.
        """

        def _tool_factory(dec_func: Union[Callable, Runnable]) -> BaseTool:
            tool_description = description
            if isinstance(dec_func, Runnable):
                runnable = dec_func

                if runnable.input_schema.model_json_schema().get("type") != "object":
                    msg = "Runnable must have an object schema."
                    raise ValueError(msg)

                async def ainvoke_wrapper(
                    callbacks: Optional[Callbacks] = None, **kwargs: Any
                ) -> Any:
                    return await runnable.ainvoke(kwargs, {"callbacks": callbacks})

                def invoke_wrapper(
                    callbacks: Optional[Callbacks] = None, **kwargs: Any
                ) -> Any:
                    return runnable.invoke(kwargs, {"callbacks": callbacks})

                coroutine = ainvoke_wrapper
                func = invoke_wrapper
                schema: Optional[ArgsSchema] = runnable.input_schema
                tool_description = description or repr(runnable)
            elif inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
                schema = args_schema
            else:
                coroutine = None
                func = dec_func
                schema = args_schema

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    description=tool_description,
                    return_direct=return_direct,
                    args_schema=schema,
                    infer_schema=infer_schema,
                    response_format=response_format,
                    parse_docstring=parse_docstring,
                    error_on_invalid_docstring=error_on_invalid_docstring,
                )
            # If someone doesn't want a schema applied, we must treat it as
            # a simple string->string function
            if dec_func.__doc__ is None:
                msg = (
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
                raise ValueError(msg)
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool",
                return_direct=return_direct,
                coroutine=coroutine,
                response_format=response_format,
            )

        return _tool_factory

    if len(args) != 0:
        # Triggered if a user attempts to use positional arguments that
        # do not exist in the function signature
        # e.g., @tool("name", runnable, "extra_arg")
        # Here, "extra_arg" is not a valid argument
        msg = "Too many arguments for tool decorator. A decorator "
        raise ValueError(msg)

    if runnable is not None:
        # tool is used as a function
        # tool_from_runnable = tool("name", runnable)
        if not name_or_callable:
            msg = "Runnable without name for tool constructor"
            raise ValueError(msg)
        if not isinstance(name_or_callable, str):
            msg = "Name must be a string for tool constructor"
            raise ValueError(msg)
        return _create_tool_factory(name_or_callable)(runnable)
    if name_or_callable is not None:
        if callable(name_or_callable) and hasattr(name_or_callable, "__name__"):
            # Used as a decorator without parameters
            # @tool
            # def my_tool():
            #    pass
            return _create_tool_factory(name_or_callable.__name__)(name_or_callable)
        if isinstance(name_or_callable, str):
            # Used with a new name for the tool
            # @tool("search")
            # def my_tool():
            #    pass
            #
            # or
            #
            # @tool("search", parse_docstring=True)
            # def my_tool():
            #    pass
            return _create_tool_factory(name_or_callable)
        msg = (
            f"The first argument must be a string or a callable with a __name__ "
            f"for tool decorator. Got {type(name_or_callable)}"
        )
        raise ValueError(msg)

    # Tool is used as a decorator with parameters specified
    # @tool(parse_docstring=True)
    # def my_tool():
    #    pass
    def _partial(func: Union[Callable, Runnable]) -> BaseTool:
        """Partial function that takes a callable and returns a tool."""
        name_ = func.get_name() if isinstance(func, Runnable) else func.__name__
        tool_factory = _create_tool_factory(name_)
        return tool_factory(func)

    return _partial


def _get_description_from_runnable(runnable: Runnable) -> str:
    """Generate a placeholder description of a runnable."""
    input_schema = runnable.input_schema.model_json_schema()
    return f"Takes {input_schema}."


def _get_schema_from_runnable_and_arg_types(
    runnable: Runnable,
    name: str,
    arg_types: Optional[dict[str, type]] = None,
) -> type[BaseModel]:
    """Infer args_schema for tool."""
    if arg_types is None:
        try:
            arg_types = get_type_hints(runnable.InputType)
        except TypeError as e:
            msg = (
                "Tool input must be str or dict. If dict, dict arguments must be "
                "typed. Either annotate types (e.g., with TypedDict) or pass "
                f"arg_types into `.as_tool` to specify. {str(e)}"
            )
            raise TypeError(msg) from e
    fields = {key: (key_type, Field(...)) for key, key_type in arg_types.items()}
    return create_model(name, **fields)  # type: ignore[call-overload]


def convert_runnable_to_tool(
    runnable: Runnable,
    args_schema: Optional[type[BaseModel]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    arg_types: Optional[dict[str, type]] = None,
) -> BaseTool:
    """Convert a Runnable into a BaseTool.

    Args:
        runnable: The runnable to convert.
        args_schema: The schema for the tool's input arguments. Defaults to None.
        name: The name of the tool. Defaults to None.
        description: The description of the tool. Defaults to None.
        arg_types: The types of the arguments. Defaults to None.

    Returns:
        The tool.
    """
    if args_schema:
        runnable = runnable.with_types(input_type=args_schema)
    description = description or _get_description_from_runnable(runnable)
    name = name or runnable.get_name()

    schema = runnable.input_schema.model_json_schema()
    if schema.get("type") == "string":
        return Tool(
            name=name,
            func=runnable.invoke,
            coroutine=runnable.ainvoke,
            description=description,
        )

    async def ainvoke_wrapper(
        callbacks: Optional[Callbacks] = None, **kwargs: Any
    ) -> Any:
        return await runnable.ainvoke(kwargs, config={"callbacks": callbacks})

    def invoke_wrapper(callbacks: Optional[Callbacks] = None, **kwargs: Any) -> Any:
        return runnable.invoke(kwargs, config={"callbacks": callbacks})

    if (
        arg_types is None
        and schema.get("type") == "object"
        and schema.get("properties")
    ):
        args_schema = runnable.input_schema
    else:
        args_schema = _get_schema_from_runnable_and_arg_types(
            runnable, name, arg_types=arg_types
        )

    return StructuredTool.from_function(
        name=name,
        func=invoke_wrapper,
        coroutine=ainvoke_wrapper,
        description=description,
        args_schema=args_schema,
    )
