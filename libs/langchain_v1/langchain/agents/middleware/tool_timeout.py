"""Tool timeout middleware for agents."""

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import override

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest


class ToolTimeoutMiddleware(AgentMiddleware[Any, Any, Any]):
    """Enforce execution timeouts on tools.

    This middleware wraps tool execution in a strict timeout. If the tool fails to
    complete within the specified duration, the execution is interrupted (or abandoned)
    and a ``ToolMessage`` with ``status="error"`` is returned natively to the LLM.

    This prevents agents from permanently freezing when interacting with unresponsive
    external APIs or hung processes.

    Examples:
        !!! example "Basic usage"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware.tool_timeout import ToolTimeoutMiddleware

            # Tools will be forcefully timed out after 5.0 seconds
            timeout_middleware = ToolTimeoutMiddleware(timeout=5.0)
            agent = create_agent("openai:gpt-4o", middleware=[timeout_middleware])
            ```
    """

    def __init__(self, timeout: float) -> None:
        """Initialize the timeout middleware.

        Args:
            timeout: Maximum allowed execution time in seconds.

        Raises:
            ValueError: If timeout is perfectly zero or negative.
        """
        super().__init__()
        if timeout <= 0:
            msg = f"Timeout must be safely greater than 0, got {timeout}."
            raise ValueError(msg)
        self.timeout = timeout

    def _generate_timeout_error(self, request: ToolCallRequest) -> ToolMessage:
        """Generate a generic ToolMessage informing the LLM of the timeout.

        Args:
            request: The current tool call request.

        Returns:
            A ToolMessage containing the timeout payload and error status.
        """
        tool_call_id = request.tool_call.get("id", "")
        name = request.tool_call.get("name", "unknown_tool")

        return ToolMessage(
            content=f"Error: Tool '{name}' execution timed out after {self.timeout} seconds.",
            name=name,
            tool_call_id=tool_call_id,
            status="error",
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap synchronous tool execution with a timeout.

        Uses a ThreadPoolExecutor to bound the execution. If the tool hangs, the
        thread is abandoned and an error message is immediately returned.

        Args:
            request: Tool call request object.
            handler: Synchronous tool handler.

        Returns:
            The tool's result, or an error ``ToolMessage`` if a timeout occurred.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(handler, request)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return self._generate_timeout_error(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap asynchronous tool execution with a timeout.

        Uses ``asyncio.wait_for`` to strictly bound the execution.

        Args:
            request: Tool call request object.
            handler: Asynchronous tool handler.

        Returns:
            The tool's result, or an error ``ToolMessage`` if a timeout occurred.
        """
        try:
            return await asyncio.wait_for(handler(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return self._generate_timeout_error(request)
