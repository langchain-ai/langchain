"""Experimental nested execution through an agent's existing tool node."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import ToolCall, ToolMessage
from langgraph.config import get_config
from langgraph.func import task
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolNode, ToolRuntime


@dataclass(frozen=True)
class ToolExecutor:
    """Experimental executor for message-returning tools, using per-call HITL."""

    node: ToolNode
    runtime: ToolRuntime
    batched_approval: bool = False

    def _input(self, call: ToolCall) -> dict[str, object]:
        if self.batched_approval:
            msg = "Nested execution requires per-call HITL, not batched HITL"
            raise ValueError(msg)
        if not call.get("id"):
            msg = "Nested tool calls require a stable ID"
            raise ValueError(msg)
        return {
            "__type": "tool_call_with_context",
            "tool_call": call,
            "state": self.runtime.state,
        }

    def _name(self, call: ToolCall) -> str:
        encoded = json.dumps(call, sort_keys=True).encode()
        return f"tool.{hashlib.sha256(encoded).hexdigest()}"

    def _runtime(self) -> Runtime:
        return Runtime(
            context=self.runtime.context,
            store=self.runtime.store,
            stream_writer=self.runtime.stream_writer,
            execution_info=self.runtime.execution_info,
            server_info=self.runtime.server_info,
        )

    def _result(self, value: object) -> ToolMessage:
        if isinstance(value, dict) and set(value) == {"messages"}:
            value = value["messages"]
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], ToolMessage):
            return value[0]
        msg = "Nested execution prototype supports ToolMessage results, not Command updates"
        raise TypeError(msg)

    def invoke(self, call: ToolCall) -> ToolMessage:
        """Execute a stable nested call through middleware and a LangGraph task."""
        payload = self._input(call)

        def execute(value: dict[str, object]) -> ToolMessage:
            result = self.node.invoke(value, get_config(), runtime=self._runtime())
            return self._result(result)

        execute.__name__ = self._name(call)
        return task(execute)(payload).result()

    async def ainvoke(self, call: ToolCall) -> ToolMessage:
        """Execute a stable nested call asynchronously through the same tool node."""
        payload = self._input(call)

        async def execute(value: dict[str, object]) -> ToolMessage:
            result = await self.node.ainvoke(value, get_config(), runtime=self._runtime())
            return self._result(result)

        execute.__name__ = self._name(call)
        return await task(execute)(payload)


_tool_executor: ContextVar[ToolExecutor] = ContextVar("langchain_tool_executor")


def get_tool_executor() -> ToolExecutor:
    """Return the experimental message-only executor; per-call HITL is required."""
    try:
        return _tool_executor.get()
    except LookupError:
        msg = "Tool execution is only available inside a create_agent tool call"
        raise RuntimeError(msg) from None
