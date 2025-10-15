"""Tool call limit middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


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


class ToolCallLimitMiddleware(AgentMiddleware):
    """Middleware that tracks tool call counts and enforces limits.

    This middleware monitors the number of tool calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware counts all tool calls in the entire message history
    and persists this count across multiple runs (invocations) of the agent.

    Run-level: The middleware counts tool calls made after the last HumanMessage,
    representing the current run (invocation) of the agent.

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
            ValueError: If both limits are None or if exit_behavior is invalid.
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
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Check tool call limits before making a model call.

        Args:
            state: The current agent state containing messages.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is "end", returns
            a Command to jump to the end with a limit exceeded message. Otherwise returns None.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and exit_behavior
                is "error".
        """
        messages = state.get("messages", [])

        # Count tool calls in entire thread
        thread_count = _count_tool_calls_in_messages(messages, self.tool_name)

        # Count tool calls in current run (after last HumanMessage)
        run_messages = _get_run_messages(messages)
        run_count = _count_tool_calls_in_messages(run_messages, self.tool_name)

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
