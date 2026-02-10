"""Utilities for injecting runtime arguments into tool function calls.

This module provides shared logic for injecting arguments like tool_call_id
and run_id into tool functions at runtime, ensuring that tools with injected
arguments work correctly across all execution paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig


def inject_runtime_args(
    func: Callable[..., Any],
    injected_keys: frozenset[str],
    kwargs: dict[str, Any],
    *,
    tool_call_id: str | None = None,
    run_id: str | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Inject runtime arguments into tool function kwargs.

    This function ensures that tools with InjectedToolCallId, InjectedRunId,
    or other injected arguments receive those values at runtime, regardless
    of the execution path (invoke, run, direct call, etc.).

    Args:
        func: The tool function to be called.
        injected_keys: Set of parameter names that should be injected.
        kwargs: Current keyword arguments for the function call.
        tool_call_id: The tool call ID to inject if needed.
        run_id: The run ID to inject if needed.
        config: The runnable config to inject if needed.

    Returns:
        Updated kwargs dict with injected arguments added.

    Raises:
        ValueError: If a required injected argument cannot be provided.

    Example:
        >>> from typing import Annotated
        >>> from langchain_core.tools import InjectedToolCallId
        >>>
        >>> def my_tool(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        ...     return f"Result for {x} (call_id: {tool_call_id})"
        >>>
        >>> injected_keys = frozenset(["tool_call_id"])
        >>> kwargs = {"x": 42}
        >>> updated_kwargs = inject_runtime_args(
        ...     my_tool,
        ...     injected_keys,
        ...     kwargs,
        ...     tool_call_id="call_123"
        ... )
        >>> # updated_kwargs now contains {"x": 42, "tool_call_id": "call_123"}
    """
    # Make a copy to avoid mutating the input
    result = kwargs.copy()

    for key in injected_keys:
        # Skip if already present in kwargs
        if key in result:
            continue

        # Inject based on parameter name
        if key == "tool_call_id":
            if tool_call_id is None:
                msg = (
                    f"Tool function {func.__name__!r} requires 'tool_call_id' "
                    "to be injected, but no tool_call_id was provided. "
                    "Tools with InjectedToolCallId must be invoked with a ToolCall "
                    "that includes an 'id' field."
                )
                raise ValueError(msg)
            result[key] = tool_call_id

        elif key == "run_id":
            if run_id is None:
                msg = (
                    f"Tool function {func.__name__!r} requires 'run_id' "
                    "to be injected, but no run_id was provided."
                )
                raise ValueError(msg)
            result[key] = run_id

        elif key == "config":
            if config is None:
                msg = (
                    f"Tool function {func.__name__!r} requires 'config' "
                    "to be injected, but no config was provided."
                )
                raise ValueError(msg)
            result[key] = config

        # Add more injected args here as needed

    return result


def validate_injected_args_present(
    func: Callable[..., Any],
    injected_keys: frozenset[str],
    kwargs: dict[str, Any],
    execution_context: Literal["invoke", "run", "_run"] = "invoke",
) -> None:
    """Validate that all required injected arguments are present in kwargs.

    This is a defensive check to ensure that injected arguments have been
    properly added to kwargs before the tool function is called.

    Args:
        func: The tool function about to be called.
        injected_keys: Set of parameter names that should be injected.
        kwargs: The keyword arguments that will be passed to the function.
        execution_context: Where this validation is being called from.

    Raises:
        RuntimeError: If a required injected argument is missing from kwargs.

    Note:
        This is a defensive measure. If this error is raised, it indicates
        a bug in the injection logic, not user error.
    """
    missing = injected_keys - kwargs.keys()
    if missing:
        msg = (
            f"BUG: Tool function {func.__name__!r} requires injected arguments "
            f"{missing} but they are missing from kwargs at {execution_context}. "
            f"This indicates a bug in the tool execution path. "
            f"Current kwargs: {list(kwargs.keys())}"
        )
        raise RuntimeError(msg)
