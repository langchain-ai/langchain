from __future__ import annotations

import textwrap
from collections.abc import Awaitable
from inspect import signature
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, Field, SkipValidation

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig, run_in_executor
from langchain_core.tools.base import (
    FILTERED_ARGS,
    BaseTool,
    _get_runnable_config_param,
    create_schema_from_function,
)
from langchain_core.utils.pydantic import TypeBaseModel


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Annotated[TypeBaseModel, SkipValidation()] = Field(
        ..., description="The tool schema."
    )
    """The input arguments' schema."""
    func: Optional[Callable[..., Any]] = None
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""

    # --- Runnable ---

    # TODO: Is this needed?
    async def ainvoke(
        self,
        input: Union[str, dict, ToolCall],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
            # If the tool does not implement async, fall back to default implementation
            return await run_in_executor(config, self.invoke, input, config, **kwargs)

        return await super().ainvoke(input, config, **kwargs)

    # --- Tool ---

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        return self.args_schema.model_json_schema()["properties"]

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
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
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously."""
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
        func: Optional[Callable] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[type[BaseModel]] = None,
        infer_schema: bool = True,
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
                Defaults to False.
            args_schema: The schema of the tool's input arguments. Defaults to None.
            infer_schema: Whether to infer the schema from the function's signature.
                Defaults to True.
            response_format: The tool response format. If "content" then the output of
                the tool is interpreted as the contents of a ToolMessage. If
                "content_and_artifact" then the output is expected to be a two-tuple
                corresponding to the (content, artifact) of a ToolMessage.
                Defaults to "content".
            parse_docstring: if ``infer_schema`` and ``parse_docstring``, will attempt
                to parse parameter descriptions from Google Style function docstrings.
                Defaults to False.
            error_on_invalid_docstring: if ``parse_docstring`` is provided, configure
                whether to raise ValueError on invalid Google Style docstrings.
                Defaults to False.
            kwargs: Additional arguments to pass to the tool

        Returns:
            The tool.

        Raises:
            ValueError: If the function is not provided.

        Examples:

            .. code-block:: python

                def add(a: int, b: int) -> int:
                    \"\"\"Add two numbers\"\"\"
                    return a + b
                tool = StructuredTool.from_function(add)
                tool.run(1, 2) # 3
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
            description_ = args_schema.__doc__ or None
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
            args_schema=args_schema,  # type: ignore[arg-type]
            description=description_,
            return_direct=return_direct,
            response_format=response_format,
            **kwargs,
        )


def _filter_schema_args(func: Callable) -> list[str]:
    filter_args = list(FILTERED_ARGS)
    if config_param := _get_runnable_config_param(func):
        filter_args.append(config_param)
    # filter_args.extend(_get_non_model_params(type_hints))
    return filter_args
