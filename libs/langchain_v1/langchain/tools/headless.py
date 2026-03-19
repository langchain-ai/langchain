"""Headless tools: schema-only tools that interrupt for out-of-process execution.

Mirrors the LangChain.js `tool` overload from
https://github.com/langchain-ai/langchainjs/pull/10430 — tools defined with
`name`, `description`, and `args_schema` only. When invoked inside a LangGraph
agent, execution pauses with an interrupt payload so a client can run the
implementation and resume.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable  # noqa: TC003
from typing import Annotated, Any, Literal, cast, overload

from langchain_core.runnables import Runnable, RunnableConfig  # noqa: TC002
from langchain_core.tools import tool as core_tool
from langchain_core.tools.base import ArgsSchema, BaseTool, InjectedToolCallId
from langchain_core.tools.structured import StructuredTool
from langchain_core.utils.pydantic import is_basemodel_subclass, is_pydantic_v2_subclass
from langgraph.types import interrupt
from pydantic import BaseModel, create_model

# Metadata flag for clients that need to detect headless tools (e.g. SDKs).
HEADLESS_TOOL_METADATA_KEY = "headless_tool"


def _args_schema_with_injected_tool_call_id(
    tool_name: str,
    args_schema: type[BaseModel],
) -> type[BaseModel]:
    """Extend a user args model with an injected `tool_call_id` field.

    The field is stripped from the model-facing tool schema but populated at
    invocation time so interrupt payloads can include the tool call id.

    Args:
        tool_name: Base name for the generated schema type.
        args_schema: Original Pydantic model for tool arguments.

    Returns:
        A new model type including `tool_call_id` injection.
    """
    model_name = f"{tool_name}HeadlessInput"
    return create_model(
        model_name,
        __base__=args_schema,
        tool_call_id=(
            Annotated[str | None, InjectedToolCallId],
            None,
        ),
    )


def _make_headless_coroutine(tool_name: str) -> Callable[..., Awaitable[Any]]:
    async def _headless_coroutine(
        _config: RunnableConfig,
        **kwargs: Any,
    ) -> Any:
        tool_call_id = kwargs.pop("tool_call_id", None)
        return interrupt(
            {
                "type": "tool",
                "tool_call": {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": kwargs,
                },
            }
        )

    return _headless_coroutine


class HeadlessTool(StructuredTool):
    """Structured tool that interrupts instead of executing locally."""


def create_headless_tool(
    *,
    name: str,
    description: str,
    args_schema: ArgsSchema,
    return_direct: bool = False,
    response_format: Literal["content", "content_and_artifact"] = "content",
    extras: dict[str, Any] | None = None,
) -> HeadlessTool:
    """Create a headless tool from a name, description, and argument schema.

    Args:
        name: Tool name exposed to the model.
        description: Tool description exposed to the model.
        args_schema: Pydantic model or JSON-schema dict for arguments.
        return_direct: Whether the tool result should end the agent turn.
        response_format: Same as `StructuredTool`.
        extras: Optional provider-specific extras merged into tool extras.

    Returns:
        A `HeadlessTool` whose coroutine raises a LangGraph interrupt when run.

    Raises:
        TypeError: If `args_schema` is not a Pydantic model or dict.
    """
    if isinstance(args_schema, dict):
        schema_for_tool: ArgsSchema = args_schema
    elif is_basemodel_subclass(args_schema):
        if is_pydantic_v2_subclass(args_schema):
            schema_for_tool = _args_schema_with_injected_tool_call_id(name, args_schema)
        else:
            schema_for_tool = args_schema
    else:
        msg = "args_schema must be a Pydantic BaseModel subclass or a dict schema."
        raise TypeError(msg)

    metadata = {HEADLESS_TOOL_METADATA_KEY: True}
    coroutine = _make_headless_coroutine(name)
    return HeadlessTool(
        name=name,
        func=None,
        coroutine=coroutine,
        description=description,
        args_schema=schema_for_tool,
        return_direct=return_direct,
        response_format=response_format,
        metadata=metadata,
        extras=extras,
    )


@overload
def tool(
    *,
    name: str,
    description: str,
    args_schema: ArgsSchema,
    return_direct: bool = False,
    response_format: Literal["content", "content_and_artifact"] = "content",
    extras: dict[str, Any] | None = None,
) -> HeadlessTool: ...


@overload
def tool(
    name_or_callable: str,
    runnable: Runnable[Any, Any],
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: Callable[..., Any],
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: str,
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]: ...


@overload
def tool(
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]: ...


def tool(
    name_or_callable: str | Callable[..., Any] | None = None,
    runnable: Runnable[Any, Any] | None = None,
    *args: Any,
    name: str | None = None,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool] | HeadlessTool:
    """Create a tool, including headless (interrupting) tools.

    If called with keyword-only `name`, `description`, and `args_schema` and no
    implementation callable, returns a `HeadlessTool` that triggers a LangGraph
    `interrupt` when executed. Otherwise delegates to `langchain_core.tools.tool`.

    Args:
        name_or_callable: Passed through to core `tool` when not using headless mode.
        runnable: Passed through to core `tool`.
        name: Tool name (headless overload only).
        description: Tool description.
        args_schema: Argument schema (`BaseModel` or JSON-schema dict).
        return_direct: Whether to return directly from the tool node.
        infer_schema: Whether to infer schema from a decorated function (core `tool`).
        response_format: Core tool response format.
        parse_docstring: Core `tool` docstring parsing flag.
        error_on_invalid_docstring: Core `tool` flag.
        extras: Optional provider-specific extras.

    Returns:
        A `HeadlessTool`, a `BaseTool`, or a decorator factory from core `tool`.
    """
    if (
        len(args) == 0
        and name_or_callable is None
        and runnable is None
        and name is not None
        and description is not None
        and args_schema is not None
    ):
        return create_headless_tool(
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            response_format=response_format,
            extras=extras,
        )
    delegated = core_tool(
        cast("Any", name_or_callable),
        cast("Any", runnable),
        *args,
        description=description,
        return_direct=return_direct,
        args_schema=args_schema,
        infer_schema=infer_schema,
        response_format=response_format,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
        extras=extras,
    )
    return cast(
        "BaseTool | Callable[[Callable[..., Any] | Runnable[Any, Any]], BaseTool]",
        delegated,
    )
