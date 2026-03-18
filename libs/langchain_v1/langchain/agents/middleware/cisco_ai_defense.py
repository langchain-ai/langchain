"""Cisco AI Defense middleware for agent security inspection.

Provides two middleware classes that inspect LLM and tool traffic against
`Cisco AI Defense <https://developer.cisco.com/docs/ai-defense/overview/>`_
security policies:

* :class:`CiscoAIDefenseMiddleware` — inspects LLM inputs and outputs via
  ``before_model`` / ``after_model`` hooks.
* :class:`CiscoAIDefenseToolMiddleware` — inspects tool call requests and
  responses via ``wrap_tool_call``.

Users compose them independently:

.. code-block:: python

    from langchain.agents import create_agent
    from langchain.agents.middleware import (
        CiscoAIDefenseMiddleware,
        CiscoAIDefenseToolMiddleware,
    )

    # LLM inspection only
    agent = create_agent(
        "openai:gpt-4.1",
        middleware=[CiscoAIDefenseMiddleware(api_key="...")],
    )

    # LLM + tool inspection
    agent = create_agent(
        "openai:gpt-4.1",
        tools=[my_tool],
        middleware=[
            CiscoAIDefenseMiddleware(api_key="..."),
            CiscoAIDefenseToolMiddleware(api_key="..."),
        ],
    )

Requires the ``aidefense`` package (``pip install aidefense``).  The import
is deferred so the package is only required at runtime, not at import time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime
    from langgraph.types import Command

logger = logging.getLogger(__name__)

_REGION_ALIASES: dict[str, str] = {
    "us": "us-west-2",
    "eu": "eu-central-1",
    "apj": "ap-northeast-1",
}


def _normalize_region(region: str) -> str:
    return _REGION_ALIASES.get(region.strip().lower(), region)


class CiscoAIDefenseError(RuntimeError):
    """Raised when Cisco AI Defense flags content and ``exit_behavior`` is ``"error"``."""

    def __init__(
        self,
        *,
        content: str,
        stage: str,
        classifications: list[str],
        message: str,
    ) -> None:
        super().__init__(message)
        self.content = content
        self.stage = stage
        self.classifications = classifications


class CiscoAIDefenseMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT],
):
    """Inspect LLM inputs and outputs against Cisco AI Defense policies.

    This middleware calls the Cisco AI Defense Chat Inspection API before and
    after every model invocation.  When a policy violation is detected the
    middleware reacts according to ``exit_behavior``:

    * ``"end"`` — jump to the end node with a violation message (default).
    * ``"error"`` — raise :class:`CiscoAIDefenseError`.

    Example:
        .. code-block:: python

            from langchain.agents import create_agent
            from langchain.agents.middleware import CiscoAIDefenseMiddleware

            agent = create_agent(
                "openai:gpt-4.1",
                middleware=[
                    CiscoAIDefenseMiddleware(
                        api_key="your-cisco-ai-defense-api-key",
                        region="us",
                        exit_behavior="end",
                    ),
                ],
            )
    """

    def __init__(
        self,
        *,
        api_key: str,
        region: str = "us",
        check_input: bool = True,
        check_output: bool = True,
        exit_behavior: Literal["end", "error"] = "end",
        fail_open: bool = True,
        timeout: int = 30,
    ) -> None:
        """Create the middleware.

        Args:
            api_key: Cisco AI Defense API key.
            region: AI Defense region — short aliases ``"us"``, ``"eu"``,
                ``"apj"`` are accepted and normalized automatically.
            check_input: Inspect user messages before the model call.
            check_output: Inspect AI messages after the model call.
            exit_behavior: How to handle violations.
                ``"end"`` jumps to the end node with a message.
                ``"error"`` raises :class:`CiscoAIDefenseError`.
            fail_open: When ``True`` (default), allow the request if the
                inspection API is unreachable.  When ``False``, treat API
                errors as violations.
            timeout: Inspection API timeout in seconds.
        """
        super().__init__()
        if exit_behavior not in ("end", "error"):
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be 'end' or 'error'."
            raise ValueError(msg)

        self.api_key = api_key
        self.region = _normalize_region(region)
        self.check_input = check_input
        self.check_output = check_output
        self.exit_behavior = exit_behavior
        self.fail_open = fail_open
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from aidefense.config import Config
            from aidefense.runtime import ChatInspectionClient

            config = Config(region=self.region, timeout=self.timeout)
            self._client = ChatInspectionClient(api_key=self.api_key, config=config)
        return self._client

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Inspect input messages before the model call.

        Args:
            state: Current agent state.
            runtime: Agent runtime context.

        Returns:
            State update to block the request, or ``None`` to continue.
        """
        if not self.check_input:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        result = self._inspect(messages)
        if result is not None and not result.is_safe:
            return self._apply_violation(messages, stage="input", result=result)
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async inspect input messages before the model call.

        Args:
            state: Current agent state.
            runtime: Agent runtime context.

        Returns:
            State update to block the request, or ``None`` to continue.
        """
        if not self.check_input:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        result = await asyncio.to_thread(self._inspect, messages)
        if result is not None and not result.is_safe:
            return self._apply_violation(messages, stage="input", result=result)
        return None

    @hook_config(can_jump_to=["end"])
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Inspect AI messages after the model call.

        Args:
            state: Current agent state.
            runtime: Agent runtime context.

        Returns:
            State update to block the response, or ``None`` to continue.
        """
        if not self.check_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        result = self._inspect(messages)
        if result is not None and not result.is_safe:
            return self._apply_violation(messages, stage="output", result=result)
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async inspect AI messages after the model call.

        Args:
            state: Current agent state.
            runtime: Agent runtime context.

        Returns:
            State update to block the response, or ``None`` to continue.
        """
        if not self.check_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        result = await asyncio.to_thread(self._inspect, messages)
        if result is not None and not result.is_safe:
            return self._apply_violation(messages, stage="output", result=result)
        return None

    # -- internals ---------------------------------------------------------

    def _inspect(self, messages: list[BaseMessage]) -> Any | None:
        """Run chat inspection with fail-open protection."""
        from aidefense.runtime.models import Message, Role

        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = Role.USER
            elif isinstance(msg, AIMessage):
                role = Role.ASSISTANT
            else:
                role = Role.SYSTEM
            converted.append(Message(role=role, content=str(msg.content)))

        try:
            return self._get_client().inspect_conversation(converted)
        except Exception:
            if self.fail_open:
                logger.warning(
                    "Cisco AI Defense inspection failed (fail_open=True), allowing request",
                    exc_info=True,
                )
                return None
            raise

    def _apply_violation(
        self,
        messages: list[BaseMessage],
        *,
        stage: str,
        result: Any,
    ) -> dict[str, Any] | None:
        classifications = (
            [str(c) for c in result.classifications] if result.classifications else []
        )
        violation_text = (
            f"This request was blocked by Cisco AI Defense ({stage} policy violation). "
            f"Classifications: {', '.join(classifications) or 'unspecified'}."
        )

        if self.exit_behavior == "error":
            last_content = str(messages[-1].content) if messages else ""
            raise CiscoAIDefenseError(
                content=last_content,
                stage=stage,
                classifications=classifications,
                message=violation_text,
            )

        return {
            "jump_to": "end",
            "messages": [AIMessage(content=violation_text)],
        }


class CiscoAIDefenseToolMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT],
):
    """Inspect tool call requests and responses against Cisco AI Defense policies.

    This middleware wraps every tool call with pre-call (request) and post-call
    (response) inspection via the Cisco AI Defense MCP Inspection API.

    It covers all tool types uniformly — LangChain ``@tool`` functions, MCP
    tools, and any tool executed through the agent's tool node.

    Example:
        .. code-block:: python

            from langchain.agents import create_agent
            from langchain.agents.middleware import CiscoAIDefenseToolMiddleware

            agent = create_agent(
                "openai:gpt-4.1",
                tools=[my_tool],
                middleware=[
                    CiscoAIDefenseToolMiddleware(
                        api_key="your-cisco-ai-defense-api-key",
                        region="us",
                    ),
                ],
            )
    """

    def __init__(
        self,
        *,
        api_key: str,
        region: str = "us",
        inspect_requests: bool = True,
        inspect_responses: bool = True,
        exit_behavior: Literal["end", "error"] = "end",
        fail_open: bool = True,
        timeout: int = 30,
    ) -> None:
        """Create the tool inspection middleware.

        Args:
            api_key: Cisco AI Defense API key.
            region: AI Defense region — ``"us"``, ``"eu"``, ``"apj"`` accepted.
            inspect_requests: Inspect tool call arguments before execution.
            inspect_responses: Inspect tool results after execution.
            exit_behavior: ``"end"`` returns a blocking ToolMessage.
                ``"error"`` raises :class:`CiscoAIDefenseError`.
            fail_open: Allow the tool call when the inspection API is
                unreachable.
            timeout: Inspection API timeout in seconds.
        """
        super().__init__()
        if exit_behavior not in ("end", "error"):
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be 'end' or 'error'."
            raise ValueError(msg)

        self.api_key = api_key
        self.region = _normalize_region(region)
        self.inspect_requests = inspect_requests
        self.inspect_responses = inspect_responses
        self.exit_behavior = exit_behavior
        self.fail_open = fail_open
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from aidefense.config import Config
            from aidefense.runtime import MCPInspectionClient

            config = Config(region=self.region, timeout=self.timeout)
            self._client = MCPInspectionClient(api_key=self.api_key, config=config)
        return self._client

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Inspect tool calls before and after execution.

        Args:
            request: Tool call request.
            handler: Callback that executes the tool.

        Returns:
            Tool result or a blocking ``ToolMessage`` on violation.
        """
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        tool_call_id = str(request.tool_call.get("id", f"blocked-{tool_name}"))

        if self.inspect_requests:
            result = self._inspect_tool_call(tool_name, tool_args)
            if result is not None and not self._is_safe(result):
                return self._apply_tool_violation(
                    tool_name, tool_call_id, "request", result,
                )

        tool_result = handler(request)

        if self.inspect_responses:
            result_data = self._extract_result_data(tool_result)
            if result_data is not None:
                result = self._inspect_response(tool_name, tool_args, result_data)
                if result is not None and not self._is_safe(result):
                    return self._apply_tool_violation(
                        tool_name, tool_call_id, "response", result,
                    )

        return tool_result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command[Any]:
        """Async inspect tool calls before and after execution.

        Args:
            request: Tool call request.
            handler: Async callback that executes the tool.

        Returns:
            Tool result or a blocking ``ToolMessage`` on violation.
        """
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        tool_call_id = str(request.tool_call.get("id", f"blocked-{tool_name}"))

        if self.inspect_requests:
            result = await asyncio.to_thread(
                self._inspect_tool_call, tool_name, tool_args,
            )
            if result is not None and not self._is_safe(result):
                return self._apply_tool_violation(
                    tool_name, tool_call_id, "request", result,
                )

        tool_result = await handler(request)

        if self.inspect_responses:
            result_data = self._extract_result_data(tool_result)
            if result_data is not None:
                result = await asyncio.to_thread(
                    self._inspect_response, tool_name, tool_args, result_data,
                )
                if result is not None and not self._is_safe(result):
                    return self._apply_tool_violation(
                        tool_name, tool_call_id, "response", result,
                    )

        return tool_result

    # -- internals ---------------------------------------------------------

    def _inspect_tool_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> Any | None:
        try:
            return self._get_client().inspect_tool_call(
                tool_name=tool_name, arguments=arguments,
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "Cisco AI Defense tool inspection failed (fail_open=True), "
                    "allowing tool call: %s",
                    tool_name,
                    exc_info=True,
                )
                return None
            raise

    def _inspect_response(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_data: dict[str, Any],
    ) -> Any | None:
        try:
            return self._get_client().inspect_response(
                result_data=result_data,
                method="tools/call",
                params={"name": tool_name, "arguments": arguments},
            )
        except Exception:
            if self.fail_open:
                logger.warning(
                    "Cisco AI Defense tool response inspection failed "
                    "(fail_open=True), allowing result: %s",
                    tool_name,
                    exc_info=True,
                )
                return None
            raise

    @staticmethod
    def _is_safe(result: Any) -> bool:
        if hasattr(result, "result") and result.result is not None:
            return result.result.is_safe
        if hasattr(result, "error") and result.error is not None:
            return False
        return True

    def _apply_tool_violation(
        self,
        tool_name: str,
        tool_call_id: str,
        direction: str,
        result: Any,
    ) -> ToolMessage:
        violation_text = (
            f"Tool call '{tool_name}' was blocked by Cisco AI Defense "
            f"({direction} policy violation)."
        )

        if self.exit_behavior == "error":
            raise CiscoAIDefenseError(
                content=f"{tool_name}({direction})",
                stage=f"tool_{direction}",
                classifications=[],
                message=violation_text,
            )

        return ToolMessage(content=violation_text, tool_call_id=tool_call_id)

    @staticmethod
    def _extract_result_data(
        tool_result: ToolMessage | Command[Any],
    ) -> dict[str, Any] | None:
        if isinstance(tool_result, ToolMessage):
            content = tool_result.content
            if isinstance(content, str):
                return {"content": [{"type": "text", "text": content}]}
            if isinstance(content, dict):
                return content
            return {"content": str(content)}
        return None


__all__ = [
    "CiscoAIDefenseError",
    "CiscoAIDefenseMiddleware",
    "CiscoAIDefenseToolMiddleware",
]
