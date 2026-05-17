"""Compensation middleware for transactional tool rollback."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ToolCallRequest,
)


class CompensationMiddleware(AgentMiddleware):
    """Middleware providing transactional compensation for tool calls.

    When a tool call succeeds, a compensation entry is added to the recovery log.
    If a later tool call fails, compensating tools are executed in reverse order.
    """

    def __init__(
        self,
        compensation_pairs: dict[str, str],
        compensation_schemas: dict[str, Callable[[Any], dict[str, Any]]],
        *,
        state_key: str = "recovery_log",
    ) -> None:
        """Initialize compensation middleware."""
        self.compensation_pairs = compensation_pairs
        self.compensation_schemas = compensation_schemas
        self.state_key = state_key

    # Get the history of suceeded tool calls
    def _get_recovery_log(
        self,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        log = state.get(self.state_key)

        if log is None:
            log = []
            state[self.state_key] = log

        return log

    # Append the history of suceeded tool calls
    def _append_recovery_entry(
        self,
        *,
        recovery_log: list[dict[str, Any]],
        tool_name: str,
        compensation_tool: str,
        compensation_args: dict[str, Any],
    ) -> None:
        recovery_log.append(
            {
                "tool": tool_name,
                "compensation_tool": compensation_tool,
                "compensation_args": compensation_args,
            }
        )

    # Build the compensation entry (tool name and arguments)
    def _build_compensation_entry(
        self,
        *,
        tool_name: str,
        result: Any,
    ) -> dict[str, Any] | None:
        if tool_name not in self.compensation_pairs:
            return None

        compensation_tool = self.compensation_pairs[tool_name]

        schema_builder = self.compensation_schemas.get(tool_name)

        compensation_args = {} if schema_builder is None else schema_builder(result)
        return {
            "compensation_tool": compensation_tool,
            "compensation_args": compensation_args,
        }

    # Execute the compensation tools in reverse order of suceeded tool calls
    def _execute_compensations_sync(
        self,
        *,
        request: ToolCallRequest,
        recovery_log: list[dict[str, Any]],
    ) -> None:
        tools = request.runtime.tools

        for entry in reversed(recovery_log):
            compensation_tool_name = entry["compensation_tool"]
            compensation_args = entry["compensation_args"]

            compensation_tool = tools.get(compensation_tool_name)

            if compensation_tool is None:
                continue

            with suppress(Exception):
                compensation_tool.invoke(compensation_args)

    async def _execute_compensations_async(
        self,
        *,
        request: ToolCallRequest,
        recovery_log: list[dict[str, Any]],
    ) -> None:
        tools = request.runtime.tools

        for entry in reversed(recovery_log):
            compensation_tool_name = entry["compensation_tool"]
            compensation_args = entry["compensation_args"]

            compensation_tool = tools.get(compensation_tool_name)

            if compensation_tool is None:
                continue

            with suppress(Exception):
                await compensation_tool.ainvoke(compensation_args)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Wrap synchronous tool execution with rollback support."""
        recovery_log = self._get_recovery_log(request.state)

        tool_name = request.tool_call["name"]

        try:
            result = handler(request)

            compensation_entry = self._build_compensation_entry(
                tool_name=tool_name,
                result=result,
            )

            if compensation_entry is not None:
                self._append_recovery_entry(
                    recovery_log=recovery_log,
                    tool_name=tool_name,
                    compensation_tool=compensation_entry["compensation_tool"],
                    compensation_args=compensation_entry["compensation_args"],
                )

        except Exception:
            self._execute_compensations_sync(
                request=request,
                recovery_log=recovery_log,
            )
            raise
        else:
            return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Wrap asynchronous tool execution with rollback support."""
        recovery_log = self._get_recovery_log(request.state)

        tool_name = request.tool_call["name"]

        try:
            result = await handler(request)

            compensation_entry = self._build_compensation_entry(
                tool_name=tool_name,
                result=result,
            )

            if compensation_entry is not None:
                self._append_recovery_entry(
                    recovery_log=recovery_log,
                    tool_name=tool_name,
                    compensation_tool=compensation_entry["compensation_tool"],
                    compensation_args=compensation_entry["compensation_args"],
                )

        except Exception:
            await self._execute_compensations_async(
                request=request,
                recovery_log=recovery_log,
            )
            raise
        else:
            return result
