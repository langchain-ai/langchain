"""Tool call limit middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ToolCallLimitState(AgentState):
    """State schema for ToolCallLimitMiddleware.

    Extends AgentState with tool call tracking fields.

    The count fields are dictionaries mapping tool names to execution counts.
    This allows multiple middleware instances to track different tools independently.
    The special key "__all__" is used for tracking all tool calls globally.
    """

    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]


def _count_tool_calls_in_messages(messages: list[AnyMessage], tool_name: str | None = None) -> int:
    """Count tool calls in a list of messages.

    Args:
        messages: List of messages to count tool calls in.
        tool_name: If specified, only count calls to this specific tool.
            If `None`, count all tool calls.

    Returns:
        The total number of tool calls (optionally filtered by tool_name).
    """
    count = 0
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            if tool_name is None:
                # Count all tool calls
                count += len(message.tool_calls)
            else:
                # Count only calls to the specified tool
                count += sum(1 for tc in message.tool_calls if tc["name"] == tool_name)
    return count


def _get_run_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Get messages from the current run (after the last HumanMessage).

    Args:
        messages: Full list of messages.

    Returns:
        Messages from the current run (after last HumanMessage).
    """
    # Find the last HumanMessage
    last_human_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_index = i
            break

    # If no HumanMessage found, return all messages
    if last_human_index == -1:
        return messages

    # Return messages after the last HumanMessage
    return messages[last_human_index + 1 :]


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
    tool_desc = f"'{tool_name}' tool call" if tool_name else "Tool call"
    exceeded_limits = []
    if thread_limit is not None and thread_count >= thread_limit:
        exceeded_limits.append(f"thread limit ({thread_count}/{thread_limit})")
    if run_limit is not None and run_count >= run_limit:
        exceeded_limits.append(f"run limit ({run_count}/{run_limit})")

    return f"{tool_desc} limits exceeded: {', '.join(exceeded_limits)}"


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


class ToolCallLimitMiddleware(AgentMiddleware[ToolCallLimitState, Any]):
    """Middleware that tracks tool call counts and enforces limits.

    This middleware monitors the number of tool calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware tracks the total number of tool calls and persists
    call count across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the number of tool calls made during a single
    run (invocation) of the agent.

    Example:
        ```python
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
        from langchain.agents import create_agent

        # Limit all tool calls globally
        global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10, exit_behavior="end")

        # Limit a specific tool
        search_limiter = ToolCallLimitMiddleware(
            tool_name="search", thread_limit=5, run_limit=3, exit_behavior="end"
        )

        # Use both in the same agent
        agent = create_agent("openai:gpt-4o", middleware=[global_limiter, search_limiter])

        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    """

    state_schema = ToolCallLimitState

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
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
    def before_model(self, state: ToolCallLimitState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
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
        thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            if self.exit_behavior == "error":
                raise ToolCallLimitExceededError(
                    thread_count=thread_count,
                    run_count=run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                    tool_name=self.tool_name,
                )
            if self.exit_behavior == "end":
                # Create a message indicating the limit was exceeded
                limit_message = _build_tool_limit_exceeded_message(
                    thread_count, run_count, self.thread_limit, self.run_limit, self.tool_name
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    def after_model(self, state: ToolCallLimitState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
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
