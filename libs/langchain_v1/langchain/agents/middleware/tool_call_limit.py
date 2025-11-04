"""Tool call limit middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, TypeVar

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.typing import ContextT
from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command


ResponseT = TypeVar("ResponseT")


class ToolCallLimitState(AgentState[ResponseT], Generic[ResponseT]):
    """State schema for ToolCallLimitMiddleware.

    Extends AgentState with tool call tracking fields.

    The count fields are dictionaries mapping tool names to execution counts.
    This allows multiple middleware instances to track different tools independently.
    The special key "__all__" is used for tracking all tool calls globally.
    """

    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]


def _build_tool_limit_exceeded_message(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
    tool_name: str | None,
) -> str:
    """Build a message indicating which tool call limits were exceeded.

    Args:
        thread_count: Current thread tool call count.
        run_count: Current run tool call count.
        thread_limit: Thread tool call limit (if set).
        run_limit: Run tool call limit (if set).
        tool_name: Tool name being limited (if specific tool), or None for all tools.

    Returns:
        A formatted message describing which limits were exceeded.
    """
    tool_desc = f"'{tool_name}' tool" if tool_name else "Tool"
    exceeded_limits = []

    if thread_limit is not None and thread_count > thread_limit:
        exceeded_limits.append(f"thread limit exceeded ({thread_count}/{thread_limit} calls)")
    if run_limit is not None and run_count > run_limit:
        exceeded_limits.append(f"run limit exceeded ({run_count}/{run_limit} calls)")

    limits_text = " and ".join(exceeded_limits)

    # Build a more detailed message
    msg_parts = [
        f"{tool_desc} call limit reached.",
        f"{limits_text.capitalize()}.",
    ]

    if tool_name:
        msg_parts.append(
            f"The '{tool_name}' tool has been called too many times and cannot be "
            "executed at this time. Consider reducing the number of calls to this tool "
            "or adjusting the limits."
        )
    else:
        msg_parts.append(
            "Too many tool calls have been made. Consider reducing the number of tool "
            "calls or adjusting the limits."
        )

    return " ".join(msg_parts)


class ToolCallLimitExceededError(Exception):
    """Exception raised when tool call limits are exceeded.

    This exception is raised when the configured exit behavior is 'error'
    and either the thread or run tool call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the exception with call count information.

        Args:
            thread_count: Current thread tool call count.
            run_count: Current run tool call count.
            thread_limit: Thread tool call limit (if set).
            run_limit: Run tool call limit (if set).
            tool_name: Tool name being limited (if specific tool), or None for all tools.
        """
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_name = tool_name

        msg = _build_tool_limit_exceeded_message(
            thread_count, run_count, thread_limit, run_limit, tool_name
        )
        super().__init__(msg)


class ToolCallLimitMiddleware(
    AgentMiddleware[ToolCallLimitState[ResponseT], ContextT],
    Generic[ResponseT, ContextT],
):
    """Tracks tool call counts and enforces limits.

    This middleware monitors the number of tool calls made during agent execution
    and can terminate the agent when specified limits are exceeded. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware tracks the total number of tool calls and persists
    call count across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the number of tool calls made during a single
    run (invocation) of the agent.

    When `allow_other_tools=True` (default), the middleware allows other tools to
    continue executing even when one tool has exceeded its limit. It injects artificial
    error messages for tool calls that have exceeded their limit, while allowing other
    tool calls to proceed normally.

    When `allow_other_tools=False`, the middleware ends execution immediately when
    any limit is exceeded, similar to the traditional behavior.

    Example:
        ```python
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
        from langchain.agents import create_agent

        # Limit all tool calls globally, continue with other tools when limit exceeded
        global_limiter = ToolCallLimitMiddleware(
            thread_limit=20, run_limit=10, exit_behavior="end", allow_other_tools=True
        )

        # Limit a specific tool, end immediately when exceeded
        search_limiter = ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
            exit_behavior="end",
            allow_other_tools=False,
        )

        # Use both in the same agent
        agent = create_agent("openai:gpt-4o", middleware=[global_limiter, search_limiter])

        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    """

    state_schema = ToolCallLimitState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
        allow_other_tools: bool = True,
        inject_limits_in_prompt: bool = False,
    ) -> None:
        """Initialize the tool call limit middleware.

        Args:
            tool_name: Name of the specific tool to limit. If `None`, limits apply
                to all tools. Defaults to `None`.
            thread_limit: Maximum number of tool calls allowed per thread.
                None means no limit. Defaults to `None`.
            run_limit: Maximum number of tool calls allowed per run.
                None means no limit. Defaults to `None`.
            exit_behavior: What to do when limits are exceeded.
                - "end": Jump to the end of the agent execution and
                    inject an artificial AI message indicating that the limit was exceeded.
                - "error": Raise a ToolCallLimitExceededError
                Defaults to "end".
            allow_other_tools: When `True`, allows other tools (or other tool calls of the
                same tool if limit not exceeded for them) to continue executing even if this
                tool's limit is exceeded. When `False`, ends execution immediately when any
                limit is exceeded. Only applicable when `exit_behavior` is "end".
                Defaults to `True`.
            inject_limits_in_prompt: When `True`, injects information about tool call
                limits and current counts into the system prompt before each model call.
                This helps the model be aware of the limits and adjust its behavior
                accordingly. Defaults to `False`.

        Raises:
            ValueError: If both limits are `None` or if `exit_behavior` is invalid.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        if exit_behavior not in ("end", "error"):
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'end' or 'error'"
            raise ValueError(msg)

        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior
        self.allow_other_tools = allow_other_tools
        self.inject_limits_in_prompt = inject_limits_in_prompt

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Includes the tool name if specified to allow multiple instances
        of this middleware with different tool names.
        """
        base_name = self.__class__.__name__
        if self.tool_name:
            return f"{base_name}[{self.tool_name}]"
        return base_name

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check tool call limits before making a model call.

        Args:
            state: The current agent state containing tool call counts.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is "end", returns
            a Command to jump to the end with a limit exceeded message. Otherwise returns None.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and exit_behavior
                is "error".
        """
        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        thread_counts = state.get("thread_tool_call_count", {})
        run_counts = state.get("run_tool_call_count", {})

        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # Check if any limits are exceeded
        thread_limit_exceeded = self.thread_limit is not None and thread_count > self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count > self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            if self.exit_behavior == "error":
                raise ToolCallLimitExceededError(
                    thread_count=thread_count,
                    run_count=run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                    tool_name=self.tool_name,
                )
            if self.exit_behavior == "end" and not self.allow_other_tools:
                # Create a message indicating the limit was exceeded
                limit_message = _build_tool_limit_exceeded_message(
                    thread_count, run_count, self.thread_limit, self.run_limit, self.tool_name
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    def after_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Increment tool call counts after a model call (when tool calls are made).

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented tool call counts if tool calls were made.
        """
        # Get the last AIMessage to check for tool calls
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last AIMessage
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if not last_ai_message or not last_ai_message.tool_calls:
            return None

        # Count relevant tool calls (filter by tool_name if specified)
        tool_call_count = 0
        for tool_call in last_ai_message.tool_calls:
            if self.tool_name is None or tool_call["name"] == self.tool_name:
                tool_call_count += 1

        if tool_call_count == 0:
            return None

        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        # Get current counts
        thread_counts = state.get("thread_tool_call_count", {}).copy()
        run_counts = state.get("run_tool_call_count", {}).copy()

        # Increment counts for this key
        thread_counts[count_key] = thread_counts.get(count_key, 0) + tool_call_count
        run_counts[count_key] = run_counts.get(count_key, 0) + tool_call_count

        return {
            "thread_tool_call_count": thread_counts,
            "run_tool_call_count": run_counts,
        }

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool calls to filter out those that have exceeded their limit.

        Args:
            request: The tool call request.
            handler: The handler to execute the tool call.

        Returns:
            Either the result from the handler, or an artificial ToolMessage
            indicating the limit was exceeded.
        """
        # Only filter if allow_other_tools is True and exit_behavior is "end"
        if not self.allow_other_tools or self.exit_behavior != "end":
            return handler(request)

        # Check if this tool call should be filtered
        state = request.state
        tool_call = request.tool_call
        tool_name_in_call = tool_call["name"]

        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        # Only check if this tool call matches our filter
        if self.tool_name is not None and self.tool_name != tool_name_in_call:
            # This tool is not limited by this middleware
            return handler(request)

        thread_counts = state.get("thread_tool_call_count", {})
        run_counts = state.get("run_tool_call_count", {})

        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # Check if limits are exceeded
        thread_limit_exceeded = self.thread_limit is not None and thread_count > self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count > self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            # Return an artificial ToolMessage indicating the limit was exceeded
            limit_message = _build_tool_limit_exceeded_message(
                thread_count, run_count, self.thread_limit, self.run_limit, self.tool_name
            )
            return ToolMessage(
                content=limit_message,
                tool_call_id=tool_call["id"],
                status="error",
            )

        # Limit not exceeded, execute the tool normally
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call to intercept tool calls.

        Args:
            request: The tool call request.
            handler: The async handler to execute the tool call.

        Returns:
            Either the result from the handler, or an artificial ToolMessage
            indicating the limit was exceeded.
        """
        # Only filter if allow_other_tools is True and exit_behavior is "end"
        if not self.allow_other_tools or self.exit_behavior != "end":
            return await handler(request)

        # Check if this tool call should be filtered
        state = request.state
        tool_call = request.tool_call
        tool_name_in_call = tool_call["name"]

        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        # Only check if this tool call matches our filter
        if self.tool_name is not None and self.tool_name != tool_name_in_call:
            # This tool is not limited by this middleware
            return await handler(request)

        thread_counts = state.get("thread_tool_call_count", {})
        run_counts = state.get("run_tool_call_count", {})

        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # Check if limits are exceeded
        thread_limit_exceeded = self.thread_limit is not None and thread_count > self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count > self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            # Return an artificial ToolMessage indicating the limit was exceeded
            limit_message = _build_tool_limit_exceeded_message(
                thread_count, run_count, self.thread_limit, self.run_limit, self.tool_name
            )
            return ToolMessage(
                content=limit_message,
                tool_call_id=tool_call["id"],
                status="error",
            )

        # Limit not exceeded, execute the tool normally
        return await handler(request)

    def wrap_model_call(
        self,
        request: Any,  # ModelRequest from types
        handler: Callable[[Any], Any],  # ModelResponse
    ) -> Any:  # ModelCallResult
        """Wrap model call to optionally inject limit information into system prompt.

        Args:
            request: The model request.
            handler: The handler to execute the model call.

        Returns:
            The result from the handler.
        """
        if not self.inject_limits_in_prompt:
            return handler(request)

        # Get current counts from state
        state = request.state
        count_key = self.tool_name if self.tool_name else "__all__"

        thread_counts = state.get("thread_tool_call_count", {})
        run_counts = state.get("run_tool_call_count", {})

        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # Build the limit information message
        limit_info_parts = []

        if self.tool_name:
            limit_info_parts.append(f"IMPORTANT: The '{self.tool_name}' tool has usage limits:")
        else:
            limit_info_parts.append("IMPORTANT: Tool usage is limited during this conversation:")

        limit_details = []
        if self.thread_limit is not None:
            remaining_thread = max(0, self.thread_limit - thread_count)
            limit_details.append(
                f"- Thread limit: {thread_count}/{self.thread_limit} calls used, "
                f"{remaining_thread} remaining"
            )
        if self.run_limit is not None:
            remaining_run = max(0, self.run_limit - run_count)
            limit_details.append(
                f"- Run limit: {run_count}/{self.run_limit} calls used, {remaining_run} remaining"
            )

        limit_info_parts.extend(limit_details)

        if self.tool_name:
            limit_info_parts.append(
                f"Please be mindful of these limits when deciding to call the "
                f"'{self.tool_name}' tool."
            )
        else:
            limit_info_parts.append(
                "Please be mindful of these limits when deciding to call tools."
            )

        limit_info = "\n".join(limit_info_parts)

        # Append to existing system prompt
        if request.system_prompt:
            request.system_prompt = f"{request.system_prompt}\n\n{limit_info}"
        else:
            request.system_prompt = limit_info

        return handler(request)

    async def awrap_model_call(
        self,
        request: Any,  # ModelRequest from types
        handler: Callable[[Any], Any],  # Async handler returning ModelResponse
    ) -> Any:  # ModelCallResult
        """Async wrap model call to optionally inject limit information.

        Args:
            request: The model request.
            handler: The async handler to execute the model call.

        Returns:
            The result from the handler.
        """
        if not self.inject_limits_in_prompt:
            return await handler(request)

        # Get current counts from state
        state = request.state
        count_key = self.tool_name if self.tool_name else "__all__"

        thread_counts = state.get("thread_tool_call_count", {})
        run_counts = state.get("run_tool_call_count", {})

        thread_count = thread_counts.get(count_key, 0)
        run_count = run_counts.get(count_key, 0)

        # Build the limit information message
        limit_info_parts = []

        if self.tool_name:
            limit_info_parts.append(f"IMPORTANT: The '{self.tool_name}' tool has usage limits:")
        else:
            limit_info_parts.append("IMPORTANT: Tool usage is limited during this conversation:")

        limit_details = []
        if self.thread_limit is not None:
            remaining_thread = max(0, self.thread_limit - thread_count)
            limit_details.append(
                f"- Thread limit: {thread_count}/{self.thread_limit} calls used, "
                f"{remaining_thread} remaining"
            )
        if self.run_limit is not None:
            remaining_run = max(0, self.run_limit - run_count)
            limit_details.append(
                f"- Run limit: {run_count}/{self.run_limit} calls used, {remaining_run} remaining"
            )

        limit_info_parts.extend(limit_details)

        if self.tool_name:
            limit_info_parts.append(
                f"Please be mindful of these limits when deciding to call the "
                f"'{self.tool_name}' tool."
            )
        else:
            limit_info_parts.append(
                "Please be mindful of these limits when deciding to call tools."
            )

        limit_info = "\n".join(limit_info_parts)

        # Append to existing system prompt
        if request.system_prompt:
            request.system_prompt = f"{request.system_prompt}\n\n{limit_info}"
        else:
            request.system_prompt = limit_info

        return await handler(request)
