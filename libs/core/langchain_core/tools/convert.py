import inspect
from typing import Any, Callable, Dict, Literal, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, Field, create_model

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import Runnable
from langchain_core.tools.base import BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.tools.structured import StructuredTool


def tool(
    *args: Union[str, Callable, Runnable],
    return_direct: bool = False,
    args_schema: Optional[Type] = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
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
            def search_api(query: str) -> Tuple[str, dict]:
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
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(dec_func: Union[Callable, Runnable]) -> BaseTool:
            if isinstance(dec_func, Runnable):
                runnable = dec_func

                if runnable.input_schema.model_json_schema().get("type") != "object":
                    raise ValueError("Runnable must have an object schema.")

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
                schema: Optional[Type[BaseModel]] = runnable.input_schema
                description = repr(runnable)
            elif inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
                schema = args_schema
                description = None
            else:
                coroutine = None
                func = dec_func
                schema = args_schema
                description = None

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    description=description,
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
                raise ValueError(
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool",
                return_direct=return_direct,
                coroutine=coroutine,
                response_format=response_format,
            )

        return _make_tool

    if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], Runnable):
        return _make_with_name(args[0])(args[1])
    elif len(args) == 1 and isinstance(args[0], str):
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


def _get_description_from_runnable(runnable: Runnable) -> str:
    """Generate a placeholder description of a runnable."""
    input_schema = runnable.input_schema.model_json_schema()
    return f"Takes {input_schema}."


def _get_schema_from_runnable_and_arg_types(
    runnable: Runnable,
    name: str,
    arg_types: Optional[Dict[str, Type]] = None,
) -> Type[BaseModel]:
    """Infer args_schema for tool."""
    if arg_types is None:
        try:
            arg_types = get_type_hints(runnable.InputType)
        except TypeError as e:
            raise TypeError(
                "Tool input must be str or dict. If dict, dict arguments must be "
                "typed. Either annotate types (e.g., with TypedDict) or pass "
                f"arg_types into `.as_tool` to specify. {str(e)}"
            ) from e
    fields = {key: (key_type, Field(...)) for key, key_type in arg_types.items()}
    return create_model(name, **fields)  # type: ignore


def convert_runnable_to_tool(
    runnable: Runnable,
    args_schema: Optional[Type[BaseModel]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    arg_types: Optional[Dict[str, Type]] = None,
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
    else:

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
