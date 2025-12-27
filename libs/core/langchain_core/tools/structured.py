"""Structured tool."""

from __future__ import annotations

import functools
import textwrap
from collections.abc import Awaitable, Callable
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
)

from pydantic import Field, SkipValidation
from typing_extensions import override

# Cannot move to TYPE_CHECKING as _run/_arun parameter annotations are needed at runtime
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,  # noqa: TC001
    CallbackManagerForToolRun,  # noqa: TC001
)
from langchain_core.runnables import RunnableConfig, run_in_executor
from langchain_core.tools.base import (
    _EMPTY_SET,
    FILTERED_ARGS,
    ArgsSchema,
    BaseTool,
    _get_runnable_config_param,
    _is_injected_arg_type,
    create_schema_from_function,
)
from langchain_core.utils.pydantic import is_basemodel_subclass

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Annotated[ArgsSchema, SkipValidation()] = Field(
        ..., description="The tool schema."
    )
    """The input arguments' schema."""
    func: Callable[..., Any] | None = None
    """The function to run when the tool is called."""
    coroutine: Callable[..., Awaitable[Any]] | None = None
    """The asynchronous version of the function."""

    # --- Runnable ---

    # TODO: Is this needed?
    @override
    async def ainvoke(
        self,
        input: str | dict | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
            # If the tool does not implement async, fall back to default implementation
            return await run_in_executor(config, self.invoke, input, config, **kwargs)

        return await super().ainvoke(input, config, **kwargs)

    # --- Tool ---

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool.

        Args:
            *args: Positional arguments to pass to the tool
            config: Configuration for the run
            run_manager: Optional callback manager to use for the run
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if self.func:
            if run_manager and signature(self.func).parameters.get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.func):
                kwargs[config_param] = config
            return self.func(*args, **kwargs)
        msg = "StructuredTool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously.

        Args:
            *args: Positional arguments to pass to the tool
            config: Configuration for the run
            run_manager: Optional callback manager to use for the run
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if self.coroutine:
            if run_manager and signature(self.coroutine).parameters.get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(self.coroutine):
                kwargs[config_param] = config
            return await self.coroutine(*args, **kwargs)

        # If self.coroutine is None, then this will delegate to the default
        # implementation which is expected to delegate to _run on a separate thread.
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )

    @classmethod
    def from_function(
        cls,
        func: Callable | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,  # noqa: FBT001,FBT002
        args_schema: ArgsSchema | None = None,
        infer_schema: bool = True,  # noqa: FBT001,FBT002
        *,
        response_format: Literal["content", "content_and_artifact"] = "content",
        parse_docstring: bool = False,
        error_on_invalid_docstring: bool = False,
        **kwargs: Any,
    ) -> StructuredTool:
        """Create tool from a given function.

        A classmethod that helps to create a tool from a function.

        Args:
            func: The function from which to create a tool.
            coroutine: The async function from which to create a tool.
            name: The name of the tool. Defaults to the function name.
            description: The description of the tool.
                Defaults to the function docstring.
            return_direct: Whether to return the result directly or as a callback.
            args_schema: The schema of the tool's input arguments.
            infer_schema: Whether to infer the schema from the function's signature.
            response_format: The tool response format.

                If `"content"` then the output of the tool is interpreted as the
                contents of a `ToolMessage`. If `"content_and_artifact"` then the output
                is expected to be a two-tuple corresponding to the `(content, artifact)`
                of a `ToolMessage`.
            parse_docstring: If `infer_schema` and `parse_docstring`, will attempt
                to parse parameter descriptions from Google Style function docstrings.
            error_on_invalid_docstring: if `parse_docstring` is provided, configure
                whether to raise `ValueError` on invalid Google Style docstrings.
            **kwargs: Additional arguments to pass to the tool

        Returns:
            The tool.

        Raises:
            ValueError: If the function is not provided.
            ValueError: If the function does not have a docstring and description
                is not provided.
            TypeError: If the `args_schema` is not a `BaseModel` or dict.

        Examples:
            ```python
            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers\"\"\"
                return a + b
            tool = StructuredTool.from_function(add)
            tool.run(1, 2) # 3

            ```
        """
        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            msg = "Function and/or coroutine must be provided"
            raise ValueError(msg)
        name = name or source_function.__name__
        if args_schema is None and infer_schema:
            # schema name is appended within function
            args_schema = create_schema_from_function(
                name,
                source_function,
                parse_docstring=parse_docstring,
                error_on_invalid_docstring=error_on_invalid_docstring,
                filter_args=_filter_schema_args(source_function),
            )
        description_ = description
        if description is None and not parse_docstring:
            description_ = source_function.__doc__ or None
        if description_ is None and args_schema:
            if isinstance(args_schema, type) and is_basemodel_subclass(args_schema):
                description_ = args_schema.__doc__
                if (
                    description_
                    and "A base class for creating Pydantic models" in description_
                ):
                    description_ = ""
                elif not description_:
                    description_ = None
            elif isinstance(args_schema, dict):
                description_ = args_schema.get("description")
            else:
                msg = (
                    "Invalid args_schema: expected BaseModel or dict, "
                    f"got {args_schema}"
                )
                raise TypeError(msg)
        if description_ is None:
            msg = "Function must have a docstring if description not provided."
            raise ValueError(msg)
        if description is None:
            # Only apply if using the function's docstring
            description_ = textwrap.dedent(description_).strip()

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        description_ = f"{description_.strip()}"
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=args_schema,
            description=description_,
            return_direct=return_direct,
            response_format=response_format,
            **kwargs,
        )

    @functools.cached_property
    def _injected_args_keys(self) -> frozenset[str]:
        fn = self.func or self.coroutine
        if fn is None:
            return _EMPTY_SET
        return frozenset(
            k
            for k, v in signature(fn).parameters.items()
            if _is_injected_arg_type(v.annotation)
        )


def _filter_schema_args(func: Callable) -> list[str]:
    filter_args = list(FILTERED_ARGS)
    if config_param := _get_runnable_config_param(func):
        filter_args.append(config_param)
    # filter_args.extend(_get_non_model_params(type_hints))
    return filter_args
