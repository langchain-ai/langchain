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

    This middleware implements the Saga Pattern for agent workflows. When a tool
    call executes successfully, a matching compensation blueprint is appended to a
    state-managed recovery log. If any subsequent tool call encounters an unhandled
    exception, the middleware intercepts the failure, iterates through the recovery
    log, and executes the compensating tools in reverse chronological order (LIFO).

    Attributes:
        compensation_pairs: A dictionary mapping a forward tool name to its corresponding
            compensating tool name.
        compensation_schemas: A dictionary mapping a forward tool name to a schema builder
            callable that transforms the raw tool result into compensation arguments.
        state_key: The tracking identifier key inside the `AgentState` dictionary where
            the recovery logs are recorded.
    """

    def __init__(
        self,
        compensation_pairs: dict[str, str],
        compensation_schemas: dict[str, Callable[[Any], dict[str, Any]]],
        *,
        state_key: str = "recovery_log",
    ) -> None:
        """Initialize compensation middleware.

        Args:
            compensation_pairs: Mapping from tool name to its compensation
                tool name (e.g., `{"charge_card": "refund_card"}`).
            compensation_schemas: Mapping from tool name to a callable that
                builds compensation arguments from the tool result.
            state_key: Key in agent state used to store the recovery log.
        """
        self.compensation_pairs = compensation_pairs
        self.compensation_schemas = compensation_schemas
        self.state_key = state_key

    def _get_recovery_log(
        self,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Get the recovery log from agent state, creating it if absent.

        Args:
            state: The current agent state dictionary.

        Returns:
            The list of succeeded tool call entries stored under `state_key`.
        """
        log = state.get(self.state_key)

        if log is None:
            log = []
            state[self.state_key] = log

        return log

    def _append_recovery_entry(
        self,
        *,
        recovery_log: list[dict[str, Any]],
        tool_name: str,
        compensation_tool: str,
        compensation_args: dict[str, Any],
    ) -> None:
        """Append a succeeded tool call entry to the recovery log.

        Args:
            recovery_log: The recovery log list to append to.
            tool_name: The name of the tool that succeeded.
            compensation_tool: The name of the compensation tool to run on rollback.
            compensation_args: Arguments to pass to the compensation tool.
        """
        recovery_log.append(
            {
                "tool": tool_name,
                "compensation_tool": compensation_tool,
                "compensation_args": compensation_args,
            }
        )

    def _build_compensation_entry(
        self,
        *,
        tool_name: str,
        result: Any,
    ) -> dict[str, Any] | None:
        """Build the compensation entry for a succeeded call.

        Args:
            tool_name: The name of the tool that succeeded.
            result: The return value of the tool call, used to derive compensation args.

        Returns:
            A dict with `compensation_tool` and `compensation_args` keys, or
            `None` if the tool has no registered compensation pair.
        """
        if tool_name not in self.compensation_pairs:
            return None

        compensation_tool = self.compensation_pairs[tool_name]
        schema_builder = self.compensation_schemas.get(tool_name)
        compensation_args = {} if schema_builder is None else schema_builder(result)

        return {
            "compensation_tool": compensation_tool,
            "compensation_args": compensation_args,
        }

    def _execute_compensations_sync(
        self,
        *,
        request: ToolCallRequest,
        recovery_log: list[dict[str, Any]],
    ) -> None:
        """Execute compensation tools synchronously in reverse order.

        Compensation failures are silently suppressed so they do not mask the
        original error that triggered the rollback.

        Args:
            request: The tool call request, used to access the runtime tool list.
            recovery_log: Ordered list of succeeded tool entries to compensate.
        """
        tools_by_name = {t.name: t for t in request.runtime.tools}

        for entry in reversed(recovery_log):
            compensation_tool_name = entry["compensation_tool"]
            compensation_args = entry["compensation_args"]
            compensation_tool = tools_by_name.get(compensation_tool_name)

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
        """Execute compensation tools asynchronously in reverse order.

        Compensation failures are silently suppressed so they do not mask the
        original error that triggered the rollback.

        Args:
            request: The tool call request, used to access the runtime tool list.
            recovery_log: Ordered list of succeeded tool entries to compensate.
        """
        tools_by_name = {t.name: t for t in request.runtime.tools}

        for entry in reversed(recovery_log):
            compensation_tool_name = entry["compensation_tool"]
            compensation_args = entry["compensation_args"]
            compensation_tool = tools_by_name.get(compensation_tool_name)

            if compensation_tool is None:
                continue

            with suppress(Exception):
                await compensation_tool.ainvoke(compensation_args)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Wrap synchronous tool execution with rollback support.

        On success, a compensation entry is appended to the recovery log stored
        in `request.state[state_key]`. On failure, all previously succeeded
        compensable tools are rolled back in reverse order before re-raising the
        original exception.

        Args:
            request: The tool call request containing the tool call dict, the
                current agent state, and the runtime (with the tool list).
            handler: Callable that executes the tool and returns its result.

        Returns:
            The result of the tool call on success.

        Raises:
            Exception: Re-raises the original exception from `handler` after
                executing any registered compensations.
        """
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
        """Wrap asynchronous tool execution with rollback support.

        On success, a compensation entry is appended to the recovery log stored
        in `request.state[state_key]`. On failure, all previously succeeded
        compensable tools are rolled back in reverse order before re-raising the
        original exception.

        Args:
            request: The tool call request containing the tool call dict, the
                current agent state, and the runtime (with the tool list).
            handler: Async callable that executes the tool and returns its result.

        Returns:
            The result of the tool call on success.

        Raises:
            Exception: Re-raises the original exception from `handler` after
                executing any registered compensations.
        """
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
