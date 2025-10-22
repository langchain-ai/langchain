"""Tool execution node for LangGraph workflows.

This module provides prebuilt functionality for executing tools in LangGraph.

Tools are functions that models can call to interact with external systems,
APIs, databases, or perform computations.

The module implements design patterns for:
- Parallel execution of multiple tool calls for efficiency
- Robust error handling with customizable error messages
- State injection for tools that need access to graph state
- Store injection for tools that need persistent storage
- Command-based state updates for advanced control flow

Key Components:
    `ToolNode`: Main class for executing tools in LangGraph workflows
    `InjectedState`: Annotation for injecting graph state into tools
    `InjectedStore`: Annotation for injecting persistent store into tools
    `tools_condition`: Utility function for conditional routing based on tool calls

Typical Usage:
    ```python
    from langchain_core.tools import tool
    from langchain.tools import ToolNode


    @tool
    def my_tool(x: int) -> str:
        return f"Result: {x}"


    tool_node = ToolNode([my_tool])
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as create_tool
from langchain_core.tools.base import (
    TOOL_MESSAGE_BLOCK_TYPES,
    ToolException,
    _DirectlyInjectedToolArg,
    get_all_basemodel_annotations,
)
from langgraph._internal._runnable import RunnableCallable
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.base import BaseStore  # noqa: TC002
from langgraph.types import Command, Send, StreamWriter
from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar, Unpack

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.runtime import Runtime
    from pydantic_core import ErrorDetails

# right now we use a dict as the default, can change this to AgentState, but depends
# on if this lives in LangChain or LangGraph... ideally would have some typed
# messages key
StateT = TypeVar("StateT", default=dict)
ContextT = TypeVar("ContextT", default=None)

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."
TOOL_EXECUTION_ERROR_TEMPLATE = (
    "Error executing tool '{tool_name}' with kwargs {tool_kwargs} with error:\n"
    " {error}\n"
    " Please fix the error and try again."
)
TOOL_INVOCATION_ERROR_TEMPLATE = (
    "Error invoking tool '{tool_name}' with kwargs {tool_kwargs} with error:\n"
    " {error}\n"
    " Please fix the error and try again."
)


class _ToolCallRequestOverrides(TypedDict, total=False):
    """Possible overrides for ToolCallRequest.override() method."""

    tool_call: ToolCall


@dataclass()
class ToolCallRequest:
    """Tool execution request passed to tool call interceptors.

    Attributes:
        tool_call: Tool call dict with name, args, and id from model output.
        tool: BaseTool instance to be invoked, or None if tool is not
            registered with the `ToolNode`. When tool is `None`, interceptors can
            handle the request without validation. If the interceptor calls `execute()`,
            validation will occur and raise an error for unregistered tools.
        state: Agent state (`dict`, `list`, or `BaseModel`).
        runtime: LangGraph runtime context (optional, `None` if outside graph).
    """

    tool_call: ToolCall
    tool: BaseTool | None
    state: Any
    runtime: ToolRuntime

    def override(self, **overrides: Unpack[_ToolCallRequestOverrides]) -> ToolCallRequest:
        """Replace the request with a new request with the given overrides.

        Returns a new `ToolCallRequest` instance with the specified attributes replaced.
        This follows an immutable pattern, leaving the original request unchanged.

        Args:
            **overrides: Keyword arguments for attributes to override. Supported keys:
                - tool_call: Tool call dict with name, args, and id

        Returns:
            New ToolCallRequest instance with specified overrides applied.

        Examples:
            ```python
            # Modify tool call arguments without mutating original
            modified_call = {**request.tool_call, "args": {"value": 10}}
            new_request = request.override(tool_call=modified_call)

            # Override multiple attributes
            new_request = request.override(tool_call=modified_call, state=new_state)
            ```
        """
        return replace(self, **overrides)


ToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]],
    ToolMessage | Command,
]
"""Wrapper for tool call execution with multi-call support.

Wrapper receives:
    request: ToolCallRequest with tool_call, tool, state, and runtime.
    execute: Callable to execute the tool (CAN BE CALLED MULTIPLE TIMES).

Returns:
    ToolMessage or Command (the final result).

The execute callable can be invoked multiple times for retry logic,
with potentially modified requests each time. Each call to execute
is independent and stateless.

!!! note
    When implementing middleware for `create_agent`, use
    `AgentMiddleware.wrap_tool_call` which provides properly typed
    state parameter for better type safety.

Examples:
    Passthrough (execute once):

    def handler(request, execute):
        return execute(request)

    Modify request before execution:

    ```python
    def handler(request, execute):
        request.tool_call["args"]["value"] *= 2
        return execute(request)
    ```

    Retry on error (execute multiple times):

    ```python
    def handler(request, execute):
        for attempt in range(3):
            try:
                result = execute(request)
                if is_valid(result):
                    return result
            except Exception:
                if attempt == 2:
                    raise
        return result
    ```

    Conditional retry based on response:

    ```python
    def handler(request, execute):
        for attempt in range(3):
            result = execute(request)
            if isinstance(result, ToolMessage) and result.status != "error":
                return result
            if attempt < 2:
                continue
            return result
    ```

    Cache/short-circuit without calling execute:

    ```python
    def handler(request, execute):
        if cached := get_cache(request):
            return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
        result = execute(request)
        save_cache(request, result)
        return result
    ```
"""

AsyncToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
    Awaitable[ToolMessage | Command],
]
"""Async wrapper for tool call execution with multi-call support."""


class ToolCallWithContext(TypedDict):
    """ToolCall with additional context for graph state.

    This is an internal data structure meant to help the `ToolNode` accept
    tool calls with additional context (e.g. state) when dispatched using the
    Send API.

    The Send API is used in create_agent to distribute tool calls in parallel
    and support human-in-the-loop workflows where graph execution may be paused
    for an indefinite time.
    """

    tool_call: ToolCall
    __type: Literal["tool_call_with_context"]
    """Type to parameterize the payload.

    Using "__" as a prefix to be defensive against potential name collisions with
    regular user state.
    """
    state: Any
    """The state is provided as additional context."""


def msg_content_output(output: Any) -> str | list[dict]:
    """Convert tool output to `ToolMessage` content format.

    Handles `str`, `list[dict]` (content blocks), and arbitrary objects by attempting
    JSON serialization with fallback to str().

    Args:
        output: Tool execution output of any type.

    Returns:
        String or list of content blocks suitable for `ToolMessage.content`.
    """
    if isinstance(output, str) or (
        isinstance(output, list)
        and all(isinstance(x, dict) and x.get("type") in TOOL_MESSAGE_BLOCK_TYPES for x in output)
    ):
        return output
    # Technically a list of strings is also valid message content, but it's
    # not currently well tested that all chat models support this.
    # And for backwards compatibility we want to make sure we don't break
    # any existing ToolNode usage.
    try:
        return json.dumps(output, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(output)


class ToolInvocationError(ToolException):
    """An error occurred while invoking a tool due to invalid arguments.

    This exception is only raised when invoking a tool using the `ToolNode`!
    """

    def __init__(
        self,
        tool_name: str,
        source: ValidationError,
        tool_kwargs: dict[str, Any],
        filtered_errors: list[ErrorDetails] | None = None,
    ) -> None:
        """Initialize the ToolInvocationError.

        Args:
            tool_name: The name of the tool that failed.
            source: The exception that occurred.
            tool_kwargs: The keyword arguments that were passed to the tool.
            filtered_errors: Optional list of filtered validation errors excluding
                injected arguments.
        """
        # Format error display based on filtered errors if provided
        if filtered_errors is not None:
            # Manually format the filtered errors without URLs or fancy formatting
            error_str_parts = []
            for error in filtered_errors:
                loc_str = ".".join(str(loc) for loc in error.get("loc", ()))
                msg = error.get("msg", "Unknown error")
                error_str_parts.append(f"{loc_str}: {msg}")
            error_display_str = "\n".join(error_str_parts)
        else:
            error_display_str = str(source)

        self.message = TOOL_INVOCATION_ERROR_TEMPLATE.format(
            tool_name=tool_name, tool_kwargs=tool_kwargs, error=error_display_str
        )
        self.tool_name = tool_name
        self.tool_kwargs = tool_kwargs
        self.source = source
        self.filtered_errors = filtered_errors
        super().__init__(self.message)


def _default_handle_tool_errors(e: Exception) -> str:
    """Default error handler for tool errors.

    If the tool is a tool invocation error, return its message.
    Otherwise, raise the error.
    """
    if isinstance(e, ToolInvocationError):
        return e.message
    raise e


def _handle_tool_error(
    e: Exception,
    *,
    flag: bool | str | Callable[..., str] | type[Exception] | tuple[type[Exception], ...],
) -> str:
    """Generate error message content based on exception handling configuration.

    This function centralizes error message generation logic, supporting different
    error handling strategies configured via the `ToolNode`'s `handle_tool_errors`
    parameter.

    Args:
        e: The exception that occurred during tool execution.
        flag: Configuration for how to handle the error. Can be:
            - bool: If `True`, use default error template
            - str: Use this string as the error message
            - Callable: Call this function with the exception to get error message
            - tuple: Not used in this context (handled by caller)

    Returns:
        A string containing the error message to include in the `ToolMessage`.

    Raises:
        ValueError: If flag is not one of the supported types.

    !!! note
        The tuple case is handled by the caller through exception type checking,
        not by this function directly.
    """
    if isinstance(flag, (bool, tuple)) or (isinstance(flag, type) and issubclass(flag, Exception)):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)  # type: ignore [assignment, call-arg]
    else:
        msg = (
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)
    return content


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception], ...]:
    """Infer exception types handled by a custom error handler function.

    This function analyzes the type annotations of a custom error handler to determine
    which exception types it's designed to handle. This enables type-safe error handling
    where only specific exceptions are caught and processed by the handler.

    Args:
        handler: A callable that takes an exception and returns an error message string.
                The first parameter (after self/cls if present) should be type-annotated
                with the exception type(s) to handle.

    Returns:
        A tuple of exception types that the handler can process. Returns (Exception,)
        if no specific type information is available for backward compatibility.

    Raises:
        ValueError: If the handler's annotation contains non-Exception types or
            if Union types contain non-Exception types.

    !!! note
        This function supports both single exception types and Union types for
        handlers that need to handle multiple exception types differently.
    """
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        # If it's a method, the first argument is typically 'self' or 'cls'
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(handler)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            if origin in [Union, UnionType]:
                args = get_args(first_param.annotation)
                if all(issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                msg = (
                    "All types in the error handler error annotation must be "
                    "Exception types. For example, "
                    "`def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{first_param.annotation}' instead."
                )
                raise ValueError(msg)

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            msg = (
                f"Arbitrary types are not supported in the error handler "
                f"signature. Please annotate the error with either a "
                f"specific Exception type or a union of Exception types. "
                "For example, `def custom_handler(e: ValueError)` or "
                "`def custom_handler(e: Union[ValueError, TypeError])`. "
                f"Got '{exception_type}' instead."
            )
            raise ValueError(msg)

    # If no type information is available, return (Exception,)
    # for backwards compatibility.
    return (Exception,)


def _filter_validation_errors(
    validation_error: ValidationError,
    tool_to_state_args: dict[str, str | None],
    tool_to_store_arg: str | None,
    tool_to_runtime_arg: str | None,
) -> list[ErrorDetails]:
    """Filter validation errors to only include LLM-controlled arguments.

    When a tool invocation fails validation, only errors for arguments that the LLM
    controls should be included in error messages. This ensures the LLM receives
    focused, actionable feedback about parameters it can actually fix. System-injected
    arguments (state, store, runtime) are filtered out since the LLM has no control
    over them.

    This function also removes injected argument values from the `input` field in error
    details, ensuring that only LLM-provided arguments appear in error messages.

    Args:
        validation_error: The Pydantic ValidationError raised during tool invocation.
        tool_to_state_args: Mapping of state argument names to state field names.
        tool_to_store_arg: Name of the store argument, if any.
        tool_to_runtime_arg: Name of the runtime argument, if any.

    Returns:
        List of ErrorDetails containing only errors for LLM-controlled arguments,
        with system-injected argument values removed from the input field.
    """
    injected_args = set(tool_to_state_args.keys())
    if tool_to_store_arg:
        injected_args.add(tool_to_store_arg)
    if tool_to_runtime_arg:
        injected_args.add(tool_to_runtime_arg)

    filtered_errors: list[ErrorDetails] = []
    for error in validation_error.errors():
        # Check if error location contains any injected argument
        # error['loc'] is a tuple like ('field_name',) or ('field_name', 'nested_field')
        if error["loc"] and error["loc"][0] not in injected_args:
            # Create a copy of the error dict to avoid mutating the original
            error_copy: dict[str, Any] = {**error}

            # Remove injected arguments from input_value if it's a dict
            if isinstance(error_copy.get("input"), dict):
                input_dict = error_copy["input"]
                input_copy = {k: v for k, v in input_dict.items() if k not in injected_args}
                error_copy["input"] = input_copy

            # Cast is safe because ErrorDetails is a TypedDict compatible with this structure
            filtered_errors.append(error_copy)  # type: ignore[arg-type]

    return filtered_errors


class _ToolNode(RunnableCallable):
    """A node for executing tools in LangGraph workflows.

    Handles tool execution patterns including function calls, state injection,
    persistent storage, and control flow. Manages parallel execution,
    error handling.

    Input Formats:
        1. Graph state with `messages` key that has a list of messages:
            - Common representation for agentic workflows
            - Supports custom messages key via `messages_key` parameter

        2. **Message List**: `[AIMessage(..., tool_calls=[...])]`
            - List of messages with tool calls in the last AIMessage

        3. **Direct Tool Calls**: `[{"name": "tool", "args": {...}, "id": "1", "type": "tool_call"}]`
            - Bypasses message parsing for direct tool execution
            - For programmatic tool invocation and testing

    Output Formats:
        Output format depends on input type and tool behavior:

        **For Regular tools**:
        - Dict input → `{"messages": [ToolMessage(...)]}`
        - List input → `[ToolMessage(...)]`

        **For Command tools**:
        - Returns `[Command(...)]` or mixed list with regular tool outputs
        - Commands can update state, trigger navigation, or send messages

    Args:
        tools: A sequence of tools that can be invoked by this node. Supports:
            - **BaseTool instances**: Tools with schemas and metadata
            - **Plain functions**: Automatically converted to tools with inferred schemas
        name: The name identifier for this node in the graph. Used for debugging
            and visualization. Defaults to "tools".
        tags: Optional metadata tags to associate with the node for filtering
            and organization. Defaults to `None`.
        handle_tool_errors: Configuration for error handling during tool execution.
            Supports multiple strategies:

            - **True**: Catch all errors and return a ToolMessage with the default
                error template containing the exception details.
            - **str**: Catch all errors and return a ToolMessage with this custom
                error message string.
            - **type[Exception]**: Only catch exceptions with the specified type and
                return the default error message for it.
            - **tuple[type[Exception], ...]**: Only catch exceptions with the specified
                types and return default error messages for them.
            - **Callable[..., str]**: Catch exceptions matching the callable's signature
                and return the string result of calling it with the exception.
            - **False**: Disable error handling entirely, allowing exceptions to
                propagate.

            Defaults to a callable that:
                - catches tool invocation errors (due to invalid arguments provided by the model) and returns a descriptive error message
                - ignores tool execution errors (they will be re-raised)

        messages_key: The key in the state dictionary that contains the message list.
            This same key will be used for the output `ToolMessage` objects.
            Defaults to "messages".
            Allows custom state schemas with different message field names.

    Examples:
        Basic usage:

        ```python
        from langchain.tools import ToolNode
        from langchain_core.tools import tool

        @tool
        def calculator(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        tool_node = ToolNode([calculator])
        ```

        State injection:

        ```python
        from typing_extensions import Annotated
        from langchain.tools import InjectedState

        @tool
        def context_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
            \"\"\"Some tool that uses state.\"\"\"
            return f"Query: {query}, Messages: {len(state['messages'])}"

        tool_node = ToolNode([context_tool])
        ```

        Error handling:

        ```python
        def handle_errors(e: ValueError) -> str:
            return "Invalid input provided"


        tool_node = ToolNode([my_tool], handle_tool_errors=handle_errors)
        ```
    """  # noqa: E501

    name: str = "tools"

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool
        | str
        | Callable[..., str]
        | type[Exception]
        | tuple[type[Exception], ...] = _default_handle_tool_errors,
        messages_key: str = "messages",
        wrap_tool_call: ToolCallWrapper | None = None,
        awrap_tool_call: AsyncToolCallWrapper | None = None,
    ) -> None:
        """Initialize `ToolNode` with tools and configuration.

        Args:
            tools: Sequence of tools to make available for execution.
            name: Node name for graph identification.
            tags: Optional metadata tags.
            handle_tool_errors: Error handling configuration.
            messages_key: State key containing messages.
            wrap_tool_call: Sync wrapper function to intercept tool execution. Receives
                ToolCallRequest and execute callable, returns ToolMessage or Command.
                Enables retries, caching, request modification, and control flow.
            awrap_tool_call: Async wrapper function to intercept tool execution.
                If not provided, falls back to wrap_tool_call for async execution.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self._tools_by_name: dict[str, BaseTool] = {}
        self._tool_to_state_args: dict[str, dict[str, str | None]] = {}
        self._tool_to_store_arg: dict[str, str | None] = {}
        self._tool_to_runtime_arg: dict[str, str | None] = {}
        self._handle_tool_errors = handle_tool_errors
        self._messages_key = messages_key
        self._wrap_tool_call = wrap_tool_call
        self._awrap_tool_call = awrap_tool_call
        for tool in tools:
            if not isinstance(tool, BaseTool):
                tool_ = create_tool(cast("type[BaseTool]", tool))
            else:
                tool_ = tool
            self._tools_by_name[tool_.name] = tool_
            self._tool_to_state_args[tool_.name] = _get_state_args(tool_)
            self._tool_to_store_arg[tool_.name] = _get_store_arg(tool_)
            self._tool_to_runtime_arg[tool_.name] = _get_runtime_arg(tool_)

    @property
    def tools_by_name(self) -> dict[str, BaseTool]:
        """Mapping from tool name to BaseTool instance."""
        return self._tools_by_name

    def _func(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        config_list = get_config_list(config, len(tool_calls))

        # Construct ToolRuntime instances at the top level for each tool call
        tool_runtimes = []
        for call, cfg in zip(tool_calls, config_list, strict=False):
            state = self._extract_state(input)
            tool_runtime = ToolRuntime(
                state=state,
                tool_call_id=call["id"],
                config=cfg,
                context=runtime.context,
                store=runtime.store,
                stream_writer=runtime.stream_writer,
            )
            tool_runtimes.append(tool_runtime)

        # Pass original tool calls without injection
        input_types = [input_type] * len(tool_calls)
        with get_executor_for_config(config) as executor:
            outputs = list(executor.map(self._run_one, tool_calls, input_types, tool_runtimes))

        return self._combine_tool_outputs(outputs, input_type)

    async def _afunc(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input)
        config_list = get_config_list(config, len(tool_calls))

        # Construct ToolRuntime instances at the top level for each tool call
        tool_runtimes = []
        for call, cfg in zip(tool_calls, config_list, strict=False):
            state = self._extract_state(input)
            tool_runtime = ToolRuntime(
                state=state,
                tool_call_id=call["id"],
                config=cfg,
                context=runtime.context,
                store=runtime.store,
                stream_writer=runtime.stream_writer,
            )
            tool_runtimes.append(tool_runtime)

        # Pass original tool calls without injection
        coros = []
        for call, tool_runtime in zip(tool_calls, tool_runtimes, strict=False):
            coros.append(self._arun_one(call, input_type, tool_runtime))  # type: ignore[arg-type]
        outputs = await asyncio.gather(*coros)

        return self._combine_tool_outputs(outputs, input_type)

    def _combine_tool_outputs(
        self,
        outputs: list[ToolMessage | Command],
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> list[Command | list[ToolMessage] | dict[str, list[ToolMessage]]]:
        # preserve existing behavior for non-command tool outputs for backwards
        # compatibility
        if not any(isinstance(output, Command) for output in outputs):
            # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
            return outputs if input_type == "list" else {self._messages_key: outputs}  # type: ignore[return-value, return-value]

        # LangGraph will automatically handle list of Command and non-command node
        # updates
        combined_outputs: list[Command | list[ToolMessage] | dict[str, list[ToolMessage]]] = []

        # combine all parent commands with goto into a single parent command
        parent_command: Command | None = None
        for output in outputs:
            if isinstance(output, Command):
                if (
                    output.graph is Command.PARENT
                    and isinstance(output.goto, list)
                    and all(isinstance(send, Send) for send in output.goto)
                ):
                    if parent_command:
                        parent_command = replace(
                            parent_command,
                            goto=cast("list[Send]", parent_command.goto) + output.goto,
                        )
                    else:
                        parent_command = Command(graph=Command.PARENT, goto=output.goto)
                else:
                    combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self._messages_key: [output]}
                )

        if parent_command:
            combined_outputs.append(parent_command)
        return combined_outputs

    def _execute_tool_sync(
        self,
        request: ToolCallRequest,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage | Command:
        """Execute tool call with configured error handling.

        Args:
            request: Tool execution request.
            input_type: Input format.
            config: Runnable configuration.

        Returns:
            ToolMessage or Command.

        Raises:
            Exception: If tool fails and handle_tool_errors is False.
        """
        call = request.tool_call
        tool = request.tool

        # Validate tool exists when we actually need to execute it
        if tool is None:
            if invalid_tool_message := self._validate_tool_call(call):
                return invalid_tool_message
            # This should never happen if validation works correctly
            msg = f"Tool {call['name']} is not registered with ToolNode"
            raise TypeError(msg)

        # Inject state, store, and runtime right before invocation
        injected_call = self._inject_tool_args(call, request.runtime)
        call_args = {**injected_call, "type": "tool_call"}

        try:
            try:
                response = tool.invoke(call_args, config)
            except ValidationError as exc:
                # Filter out errors for injected arguments
                filtered_errors = _filter_validation_errors(
                    exc,
                    self._tool_to_state_args.get(call["name"], {}),
                    self._tool_to_store_arg.get(call["name"]),
                    self._tool_to_runtime_arg.get(call["name"]),
                )
                # Use original call["args"] without injected values for error reporting
                raise ToolInvocationError(call["name"], exc, call["args"], filtered_errors) from exc

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios,
        # Where GraphInterrupt(GraphBubbleUp) is raised from an `interrupt` invocation
        # most commonly:
        # (1) a GraphInterrupt is raised inside a tool
        # (2) a GraphInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph
        #     called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp:
            raise
        except Exception as e:
            # Determine which exception types are handled
            handled_types: tuple[type[Exception], ...]
            if isinstance(self._handle_tool_errors, type) and issubclass(
                self._handle_tool_errors, Exception
            ):
                handled_types = (self._handle_tool_errors,)
            elif isinstance(self._handle_tool_errors, tuple):
                handled_types = self._handle_tool_errors
            elif callable(self._handle_tool_errors) and not isinstance(
                self._handle_tool_errors, type
            ):
                handled_types = _infer_handled_types(self._handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Check if this error should be handled
            if not self._handle_tool_errors or not isinstance(e, handled_types):
                raise

            # Error is handled - create error ToolMessage
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        # Process successful response
        if isinstance(response, Command):
            # Validate Command before returning to handler
            return self._validate_tool_command(response, request.tool_call, input_type)
        if isinstance(response, ToolMessage):
            response.content = cast("str | list", msg_content_output(response.content))
            return response

        msg = f"Tool {call['name']} returned unexpected type: {type(response)}"
        raise TypeError(msg)

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime: ToolRuntime,
    ) -> ToolMessage | Command:
        """Execute single tool call with wrap_tool_call wrapper if configured.

        Args:
            call: Tool call dict.
            input_type: Input format.
            tool_runtime: Tool runtime.

        Returns:
            ToolMessage or Command.
        """
        # Validation is deferred to _execute_tool_sync to allow interceptors
        # to short-circuit requests for unregistered tools
        tool = self.tools_by_name.get(call["name"])

        # Create the tool request with state and runtime
        tool_request = ToolCallRequest(
            tool_call=call,
            tool=tool,
            state=tool_runtime.state,
            runtime=tool_runtime,
        )

        config = tool_runtime.config

        if self._wrap_tool_call is None:
            # No wrapper - execute directly
            return self._execute_tool_sync(tool_request, input_type, config)

        # Define execute callable that can be called multiple times
        def execute(req: ToolCallRequest) -> ToolMessage | Command:
            """Execute tool with given request. Can be called multiple times."""
            return self._execute_tool_sync(req, input_type, config)

        # Call wrapper with request and execute callable
        try:
            return self._wrap_tool_call(tool_request, execute)
        except Exception as e:
            # Wrapper threw an exception
            if not self._handle_tool_errors:
                raise
            # Convert to error message
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=tool_request.tool_call["name"],
                tool_call_id=tool_request.tool_call["id"],
                status="error",
            )

    async def _execute_tool_async(
        self,
        request: ToolCallRequest,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage | Command:
        """Execute tool call asynchronously with configured error handling.

        Args:
            request: Tool execution request.
            input_type: Input format.
            config: Runnable configuration.

        Returns:
            ToolMessage or Command.

        Raises:
            Exception: If tool fails and handle_tool_errors is False.
        """
        call = request.tool_call
        tool = request.tool

        # Validate tool exists when we actually need to execute it
        if tool is None:
            if invalid_tool_message := self._validate_tool_call(call):
                return invalid_tool_message
            # This should never happen if validation works correctly
            msg = f"Tool {call['name']} is not registered with ToolNode"
            raise TypeError(msg)

        # Inject state, store, and runtime right before invocation
        injected_call = self._inject_tool_args(call, request.runtime)
        call_args = {**injected_call, "type": "tool_call"}

        try:
            try:
                response = await tool.ainvoke(call_args, config)
            except ValidationError as exc:
                # Filter out errors for injected arguments
                filtered_errors = _filter_validation_errors(
                    exc,
                    self._tool_to_state_args.get(call["name"], {}),
                    self._tool_to_store_arg.get(call["name"]),
                    self._tool_to_runtime_arg.get(call["name"]),
                )
                # Use original call["args"] without injected values for error reporting
                raise ToolInvocationError(call["name"], exc, call["args"], filtered_errors) from exc

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios,
        # Where GraphInterrupt(GraphBubbleUp) is raised from an `interrupt` invocation
        # most commonly:
        # (1) a GraphInterrupt is raised inside a tool
        # (2) a GraphInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph
        #     called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp:
            raise
        except Exception as e:
            # Determine which exception types are handled
            handled_types: tuple[type[Exception], ...]
            if isinstance(self._handle_tool_errors, type) and issubclass(
                self._handle_tool_errors, Exception
            ):
                handled_types = (self._handle_tool_errors,)
            elif isinstance(self._handle_tool_errors, tuple):
                handled_types = self._handle_tool_errors
            elif callable(self._handle_tool_errors) and not isinstance(
                self._handle_tool_errors, type
            ):
                handled_types = _infer_handled_types(self._handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Check if this error should be handled
            if not self._handle_tool_errors or not isinstance(e, handled_types):
                raise

            # Error is handled - create error ToolMessage
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        # Process successful response
        if isinstance(response, Command):
            # Validate Command before returning to handler
            return self._validate_tool_command(response, request.tool_call, input_type)
        if isinstance(response, ToolMessage):
            response.content = cast("str | list", msg_content_output(response.content))
            return response

        msg = f"Tool {call['name']} returned unexpected type: {type(response)}"
        raise TypeError(msg)

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime: ToolRuntime,
    ) -> ToolMessage | Command:
        """Execute single tool call asynchronously with awrap_tool_call wrapper if configured.

        Args:
            call: Tool call dict.
            input_type: Input format.
            tool_runtime: Tool runtime.

        Returns:
            ToolMessage or Command.
        """
        # Validation is deferred to _execute_tool_async to allow interceptors
        # to short-circuit requests for unregistered tools
        tool = self.tools_by_name.get(call["name"])

        # Create the tool request with state and runtime
        tool_request = ToolCallRequest(
            tool_call=call,
            tool=tool,
            state=tool_runtime.state,
            runtime=tool_runtime,
        )

        config = tool_runtime.config

        if self._awrap_tool_call is None and self._wrap_tool_call is None:
            # No wrapper - execute directly
            return await self._execute_tool_async(tool_request, input_type, config)

        # Define async execute callable that can be called multiple times
        async def execute(req: ToolCallRequest) -> ToolMessage | Command:
            """Execute tool with given request. Can be called multiple times."""
            return await self._execute_tool_async(req, input_type, config)

        def _sync_execute(req: ToolCallRequest) -> ToolMessage | Command:
            """Sync execute fallback for sync wrapper."""
            return self._execute_tool_sync(req, input_type, config)

        # Call wrapper with request and execute callable
        try:
            if self._awrap_tool_call is not None:
                return await self._awrap_tool_call(tool_request, execute)
            # None check was performed above already
            self._wrap_tool_call = cast("ToolCallWrapper", self._wrap_tool_call)
            return self._wrap_tool_call(tool_request, _sync_execute)
        except Exception as e:
            # Wrapper threw an exception
            if not self._handle_tool_errors:
                raise
            # Convert to error message
            content = _handle_tool_error(e, flag=self._handle_tool_errors)
            return ToolMessage(
                content=content,
                name=tool_request.tool_call["name"],
                tool_call_id=tool_request.tool_call["id"],
                status="error",
            )

    def _parse_input(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
    ) -> tuple[list[ToolCall], Literal["list", "dict", "tool_calls"]]:
        input_type: Literal["list", "dict", "tool_calls"]
        if isinstance(input, list):
            if isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                input_type = "tool_calls"
                tool_calls = cast("list[ToolCall]", input)
                return tool_calls, input_type
            input_type = "list"
            messages = input
        elif isinstance(input, dict) and input.get("__type") == "tool_call_with_context":
            # Handle ToolCallWithContext from Send API
            # mypy will not be able to type narrow correctly since the signature
            # for input contains dict[str, Any]. We'd need to narrow dict[str, Any]
            # before we can apply correct typing.
            input_with_ctx = cast("ToolCallWithContext", input)
            input_type = "tool_calls"
            return [input_with_ctx["tool_call"]], input_type
        elif isinstance(input, dict) and (messages := input.get(self._messages_key, [])):
            input_type = "dict"
        elif messages := getattr(input, self._messages_key, []):
            # Assume dataclass-like state that can coerce from dict
            input_type = "dict"
        else:
            msg = "No message found in input"
            raise ValueError(msg)

        try:
            latest_ai_message = next(m for m in reversed(messages) if isinstance(m, AIMessage))
        except StopIteration:
            msg = "No AIMessage found in input"
            raise ValueError(msg)

        tool_calls = list(latest_ai_message.tool_calls)
        return tool_calls, input_type

    def _validate_tool_call(self, call: ToolCall) -> ToolMessage | None:
        requested_tool = call["name"]
        if requested_tool not in self.tools_by_name:
            all_tool_names = list(self.tools_by_name.keys())
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(all_tool_names),
            )
            return ToolMessage(
                content, name=requested_tool, tool_call_id=call["id"], status="error"
            )
        return None

    def _extract_state(
        self, input: list[AnyMessage] | dict[str, Any] | BaseModel
    ) -> list[AnyMessage] | dict[str, Any] | BaseModel:
        """Extract state from input, handling ToolCallWithContext if present.

        Args:
            input: The input which may be raw state or ToolCallWithContext.

        Returns:
            The actual state to pass to wrap_tool_call wrappers.
        """
        if isinstance(input, dict) and input.get("__type") == "tool_call_with_context":
            return input["state"]
        return input

    def _inject_state(
        self,
        tool_call: ToolCall,
        state: list[AnyMessage] | dict[str, Any] | BaseModel,
    ) -> ToolCall:
        state_args = self._tool_to_state_args[tool_call["name"]]

        if state_args and isinstance(state, list):
            required_fields = list(state_args.values())
            if (
                len(required_fields) == 1 and required_fields[0] == self._messages_key
            ) or required_fields[0] is None:
                state = {self._messages_key: state}
            else:
                err_msg = (
                    f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                    f"graph state dict as input."
                )
                if any(state_field for state_field in state_args.values()):
                    required_fields_str = ", ".join(f for f in required_fields if f)
                    err_msg += f" State should contain fields {required_fields_str}."
                raise ValueError(err_msg)

        if isinstance(state, dict):
            tool_state_args = {
                tool_arg: state[state_field] if state_field else state
                for tool_arg, state_field in state_args.items()
            }
        else:
            tool_state_args = {
                tool_arg: getattr(state, state_field) if state_field else state
                for tool_arg, state_field in state_args.items()
            }

        tool_call["args"] = {
            **tool_call["args"],
            **tool_state_args,
        }
        return tool_call

    def _inject_store(self, tool_call: ToolCall, store: BaseStore | None) -> ToolCall:
        store_arg = self._tool_to_store_arg[tool_call["name"]]
        if not store_arg:
            return tool_call

        if store is None:
            msg = (
                "Cannot inject store into tools with InjectedStore annotations - "
                "please compile your graph with a store."
            )
            raise ValueError(msg)

        tool_call["args"] = {
            **tool_call["args"],
            store_arg: store,
        }
        return tool_call

    def _inject_runtime(self, tool_call: ToolCall, tool_runtime: ToolRuntime) -> ToolCall:
        """Inject ToolRuntime into tool call arguments.

        Args:
            tool_call: The tool call to inject runtime into.
            tool_runtime: The ToolRuntime instance to inject.

        Returns:
            The tool call with runtime injected if needed.
        """
        runtime_arg = self._tool_to_runtime_arg.get(tool_call["name"])
        if not runtime_arg:
            return tool_call

        tool_call["args"] = {
            **tool_call["args"],
            runtime_arg: tool_runtime,
        }
        return tool_call

    def _inject_tool_args(
        self,
        tool_call: ToolCall,
        tool_runtime: ToolRuntime,
    ) -> ToolCall:
        """Inject graph state, store, and runtime into tool call arguments.

        This is an internal method that enables tools to access graph context that
        should not be controlled by the model. Tools can declare dependencies on graph
        state, persistent storage, or runtime context using InjectedState, InjectedStore,
        and ToolRuntime annotations. This method automatically identifies these
        dependencies and injects the appropriate values.

        The injection process preserves the original tool call structure while adding
        the necessary context arguments. This allows tools to be both model-callable
        and context-aware without exposing internal state management to the model.

        Args:
            tool_call: The tool call dictionary to augment with injected arguments.
                Must contain 'name', 'args', 'id', and 'type' fields.
            tool_runtime: The ToolRuntime instance containing all runtime context
                (state, config, store, context, stream_writer) to inject into tools.

        Returns:
            A new ToolCall dictionary with the same structure as the input but with
            additional arguments injected based on the tool's annotation requirements.

        Raises:
            ValueError: If a tool requires store injection but no store is provided,
                or if state injection requirements cannot be satisfied.

        !!! note
            This method is called automatically during tool execution. It should not
            be called from outside the `ToolNode`.
        """
        if tool_call["name"] not in self.tools_by_name:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        tool_call_with_state = self._inject_state(tool_call_copy, tool_runtime.state)
        tool_call_with_store = self._inject_store(tool_call_with_state, tool_runtime.store)
        return self._inject_runtime(tool_call_with_store, tool_runtime)

    def _validate_tool_command(
        self,
        command: Command,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> Command:
        if isinstance(command.update, dict):
            # input type is dict when ToolNode is invoked with a dict input
            # (e.g. {"messages": [AIMessage(..., tool_calls=[...])]})
            if input_type not in ("dict", "tool_calls"):
                msg = (
                    "Tools can provide a dict in Command.update only when using dict "
                    f"with '{self._messages_key}' key as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )
                raise ValueError(msg)

            updated_command = deepcopy(command)
            state_update = cast("dict[str, Any]", updated_command.update) or {}
            messages_update = state_update.get(self._messages_key, [])
        elif isinstance(command.update, list):
            # Input type is list when ToolNode is invoked with a list input
            # (e.g. [AIMessage(..., tool_calls=[...])])
            if input_type != "list":
                msg = (
                    "Tools can provide a list of messages in Command.update "
                    "only when using list of messages as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )
                raise ValueError(msg)

            updated_command = deepcopy(command)
            messages_update = updated_command.update
        else:
            return command

        # convert to message objects if updates are in a dict format
        messages_update = convert_to_messages(messages_update)

        # no validation needed if all messages are being removed
        if messages_update == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]:
            return updated_command

        has_matching_tool_message = False
        for message in messages_update:
            if not isinstance(message, ToolMessage):
                continue

            if message.tool_call_id == call["id"]:
                message.name = call["name"]
                has_matching_tool_message = True

        # validate that we always have a ToolMessage matching the tool call in
        # Command.update if command is sent to the CURRENT graph
        if updated_command.graph is None and not has_matching_tool_message:
            example_update = (
                '`Command(update={"messages": '
                '[ToolMessage("Success", tool_call_id=tool_call_id), ...]}, ...)`'
                if input_type == "dict"
                else "`Command(update="
                '[ToolMessage("Success", tool_call_id=tool_call_id), ...], ...)`'
            )
            msg = (
                "Expected to have a matching ToolMessage in Command.update "
                f"for tool '{call['name']}', got: {messages_update}. "
                "Every tool call (LLM requesting to call a tool) "
                "in the message history MUST have a corresponding ToolMessage. "
                f"You can fix it by modifying the tool to return {example_update}."
            )
            raise ValueError(msg)
        return updated_command


def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Conditional routing function for tool-calling workflows.

    This utility function implements the standard conditional logic for ReAct-style
    agents: if the last AI message contains tool calls, route to the tool execution
    node; otherwise, end the workflow. This pattern is fundamental to most tool-calling
    agent architectures.

    The function handles multiple state formats commonly used in LangGraph applications,
    making it flexible for different graph designs while maintaining consistent behavior.

    Args:
        state: The current graph state to examine for tool calls. Supported formats:
            - Dictionary containing a messages key (for StateGraph)
            - BaseModel instance with a messages attribute
        messages_key: The key or attribute name containing the message list in the state.
            This allows customization for graphs using different state schemas.
            Defaults to "messages".

    Returns:
        Either "tools" if tool calls are present in the last AI message, or "__end__"
        to terminate the workflow. These are the standard routing destinations for
        tool-calling conditional edges.

    Raises:
        ValueError: If no messages can be found in the provided state format.

    Example:
        Basic usage in a ReAct agent:

        ```python
        from langgraph.graph import StateGraph
        from langchain.tools import ToolNode
        from langchain.tools.tool_node import tools_condition
        from typing_extensions import TypedDict


        class State(TypedDict):
            messages: list


        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "__end__"},
        )
        ```

        Custom messages key:

        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```

    !!! note
        This function is designed to work seamlessly with `ToolNode` and standard
        LangGraph patterns. It expects the last message to be an `AIMessage` when
        tool calls are present, which is the standard output format for tool-calling
        language models.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif (isinstance(state, dict) and (messages := state.get(messages_key, []))) or (
        messages := getattr(state, messages_key, [])
    ):
        ai_message = messages[-1]
    else:
        msg = f"No messages found in input state to tool_edge: {state}"
        raise ValueError(msg)
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


@dataclass
class ToolRuntime(_DirectlyInjectedToolArg, Generic[ContextT, StateT]):
    """Runtime context automatically injected into tools.

    When a tool function has a parameter named `tool_runtime` with type hint
    `ToolRuntime`, the tool execution system will automatically inject an instance
    containing:

    - `state`: The current graph state
    - `tool_call_id`: The ID of the current tool call
    - `config`: `RunnableConfig` for the current execution
    - `context`: Runtime context (from langgraph `Runtime`)
    - `store`: `BaseStore` instance for persistent storage (from langgraph `Runtime`)
    - `stream_writer`: `StreamWriter` for streaming output (from langgraph `Runtime`)

    No `Annotated` wrapper is needed - just use `runtime: ToolRuntime`
    as a parameter.

    Example:
        ```python
        from langchain_core.tools import tool
        from langchain.tools import ToolRuntime

        @tool
        def my_tool(x: int, runtime: ToolRuntime) -> str:
            \"\"\"Tool that accesses runtime context.\"\"\"
            # Access state
            messages = tool_runtime.state["messages"]

            # Access tool_call_id
            print(f"Tool call ID: {tool_runtime.tool_call_id}")

            # Access config
            print(f"Run ID: {tool_runtime.config.get('run_id')}")

            # Access runtime context
            user_id = tool_runtime.context.get("user_id")

            # Access store
            tool_runtime.store.put(("metrics",), "count", 1)

            # Stream output
            tool_runtime.stream_writer.write("Processing...")

            return f"Processed {x}"
        ```

    !!! note
        This is a marker class used for type checking and detection.
        The actual runtime object will be constructed during tool execution.
    """

    state: StateT
    context: ContextT
    config: RunnableConfig
    stream_writer: StreamWriter
    tool_call_id: str | None
    store: BaseStore | None


class InjectedState(InjectedToolArg):
    """Annotation for injecting graph state into tool arguments.

    This annotation enables tools to access graph state without exposing state
    management details to the language model. Tools annotated with `InjectedState`
    receive state data automatically during execution while remaining invisible
    to the model's tool-calling interface.

    Args:
        field: Optional key to extract from the state dictionary. If `None`, the entire
            state is injected. If specified, only that field's value is injected.
            This allows tools to request specific state components rather than
            processing the full state structure.

    Example:
        ```python
        from typing import List
        from typing_extensions import Annotated, TypedDict

        from langchain_core.messages import BaseMessage, AIMessage
        from langchain.tools import InjectedState, ToolNode, tool


        class AgentState(TypedDict):
            messages: List[BaseMessage]
            foo: str


        @tool
        def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
            '''Do something with state.'''
            if len(state["messages"]) > 2:
                return state["foo"] + str(x)
            else:
                return "not enough messages"


        @tool
        def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
            '''Do something else with state.'''
            return foo + str(x + 1)


        node = ToolNode([state_tool, foo_tool])

        tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
            "foo": "bar",
        }
        node.invoke(state)
        ```

        ```python
        [
            ToolMessage(content="not enough messages", name="state_tool", tool_call_id="1"),
            ToolMessage(content="bar2", name="foo_tool", tool_call_id="2"),
        ]
        ```

    !!! note
        - `InjectedState` arguments are automatically excluded from tool schemas
            presented to language models
        - `ToolNode` handles the injection process during execution
        - Tools can mix regular arguments (controlled by the model) with injected
            arguments (controlled by the system)
        - State injection occurs after the model generates tool calls but before
            tool execution
    """

    def __init__(self, field: str | None = None) -> None:
        """Initialize the `InjectedState` annotation."""
        self.field = field


class InjectedStore(InjectedToolArg):
    """Annotation for injecting persistent store into tool arguments.

    This annotation enables tools to access LangGraph's persistent storage system
    without exposing storage details to the language model. Tools annotated with
    InjectedStore receive the store instance automatically during execution while
    remaining invisible to the model's tool-calling interface.

    The store provides persistent, cross-session data storage that tools can use
    for maintaining context, user preferences, or any other data that needs to
    persist beyond individual workflow executions.

    !!! warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing_extensions import Annotated
        from langgraph.store.memory import InMemoryStore
        from langchain.tools import InjectedStore, ToolNode, tool

        @tool
        def save_preference(
            key: str,
            value: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Save user preference to persistent storage.\"\"\"
            store.put(("preferences",), key, value)
            return f"Saved {key} = {value}"

        @tool
        def get_preference(
            key: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Retrieve user preference from persistent storage.\"\"\"
            result = store.get(("preferences",), key)
            return result.value if result else "Not found"
        ```

        Usage with `ToolNode` and graph compilation:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        tool_node = ToolNode([save_preference, get_preference])

        graph = StateGraph(State)
        graph.add_node("tools", tool_node)
        compiled_graph = graph.compile(store=store)  # Store is injected automatically
        ```

        Cross-session persistence:

        ```python
        # First session
        result1 = graph.invoke({"messages": [HumanMessage("Save my favorite color as blue")]})

        # Later session - data persists
        result2 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]})
        ```

    !!! note
        - `InjectedStore` arguments are automatically excluded from tool schemas
            presented to language models
        - The store instance is automatically injected by `ToolNode` during execution
        - Tools can access namespaced storage using the store's get/put methods
        - Store injection requires the graph to be compiled with a store instance
        - Multiple tools can share the same store instance for data consistency
    """


def _is_injection(
    type_arg: Any,
    injection_type: type[InjectedState | InjectedStore | ToolRuntime],
) -> bool:
    """Check if a type argument represents an injection annotation.

    This utility function determines whether a type annotation indicates that
    an argument should be injected with state or store data. It handles both
    direct annotations and nested annotations within Union or Annotated types.

    Args:
        type_arg: The type argument to check for injection annotations.
        injection_type: The injection type to look for (InjectedState or InjectedStore).

    Returns:
        True if the type argument contains the specified injection annotation.
    """
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ is Union or origin_ is Annotated:
        return any(_is_injection(ta, injection_type) for ta in get_args(type_arg))
    return False


def _get_state_args(tool: BaseTool) -> dict[str, str | None]:
    """Extract state injection mappings from tool annotations.

    This function analyzes a tool's input schema to identify arguments that should
    be injected with graph state. It processes InjectedState annotations to build
    a mapping of tool argument names to state field names.

    Args:
        tool: The tool to analyze for state injection requirements.

    Returns:
        A dictionary mapping tool argument names to state field names. If a field
        name is None, the entire state should be injected for that argument.
    """
    full_schema = tool.get_input_schema()
    tool_args_to_state_fields: dict = {}

    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg for type_arg in get_args(type_) if _is_injection(type_arg, InjectedState)
        ]
        if len(injections) > 1:
            msg = (
                "A tool argument should not be annotated with InjectedState more than "
                f"once. Received arg {name} with annotations {injections}."
            )
            raise ValueError(msg)
        if len(injections) == 1:
            injection = injections[0]
            if isinstance(injection, InjectedState) and injection.field:
                tool_args_to_state_fields[name] = injection.field
            else:
                tool_args_to_state_fields[name] = None
        else:
            pass
    return tool_args_to_state_fields


def _get_store_arg(tool: BaseTool) -> str | None:
    """Extract store injection argument from tool annotations.

    This function analyzes a tool's input schema to identify the argument that
    should be injected with the graph store. Only one store argument is supported
    per tool.

    Args:
        tool: The tool to analyze for store injection requirements.

    Returns:
        The name of the argument that should receive the store injection, or None
        if no store injection is required.

    Raises:
        ValueError: If a tool argument has multiple InjectedStore annotations.
    """
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg for type_arg in get_args(type_) if _is_injection(type_arg, InjectedStore)
        ]
        if len(injections) > 1:
            msg = (
                "A tool argument should not be annotated with InjectedStore more than "
                f"once. Received arg {name} with annotations {injections}."
            )
            raise ValueError(msg)
        if len(injections) == 1:
            return name

    return None


def _get_runtime_arg(tool: BaseTool) -> str | None:
    """Extract runtime injection argument from tool annotations.

    This function analyzes a tool's input schema to identify the argument that
    should be injected with the ToolRuntime instance. Only one runtime argument
    is supported per tool.

    Args:
        tool: The tool to analyze for runtime injection requirements.

    Returns:
        The name of the argument that should receive the runtime injection, or None
        if no runtime injection is required.

    Raises:
        ValueError: If a tool argument has multiple ToolRuntime annotations.
    """
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        # Check if the parameter name is "runtime" (regardless of type)
        if name == "runtime":
            return name
        # Check if the type itself is ToolRuntime (direct usage)
        if _is_injection(type_, ToolRuntime):
            return name
        # Check if ToolRuntime is in Annotated args
        injections = [
            type_arg for type_arg in get_args(type_) if _is_injection(type_arg, ToolRuntime)
        ]
        if len(injections) > 1:
            msg = (
                "A tool argument should not be annotated with ToolRuntime more than "
                f"once. Received arg {name} with annotations {injections}."
            )
            raise ValueError(msg)
        if len(injections) == 1:
            return name

    return None
