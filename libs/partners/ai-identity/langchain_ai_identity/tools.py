"""AI Identity toolkit for governing LangChain tool access.

Wraps arbitrary :class:`~langchain_core.tools.BaseTool` instances with
gateway policy checks so that every tool invocation is enforced by the
AI Identity platform.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Any

from langchain_core.tools import BaseTool, ToolException

from langchain_ai_identity._gateway import (
    _DEFAULT_TIMEOUT,
    aenforce_access,
    enforce_access,
)

logger = logging.getLogger(__name__)


def _tool_endpoint(tool: BaseTool) -> str:
    """Derive the gateway endpoint string from a tool."""
    return f"/tools/{tool.name.lower().replace(' ', '_')}"


class AIIdentityToolkit:
    """Governance wrapper that enforces AI Identity policies on tools.

    Each tool passed to the toolkit is wrapped so that a gateway policy
    check runs before every invocation.  If the policy denies the call,
    a :class:`~langchain_core.tools.ToolException` is raised.

    Args:
        api_key: AI Identity API key.
        agent_id: Unique identifier for the agent.
        tools: Sequence of LangChain tools to govern.
        fail_closed: Block on deny when ``True`` (default).
        timeout: HTTP timeout for gateway calls in seconds.
        gateway_url: Override for the AI Identity gateway base URL.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        tools: list[BaseTool],
        *,
        fail_closed: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
        gateway_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.agent_id = agent_id
        self.fail_closed = fail_closed
        self.timeout = timeout
        self.gateway_url = gateway_url
        self._tools = [self._wrap_tool(t) for t in tools]

    # -- public API -----------------------------------------------------------

    def get_tools(self) -> list[BaseTool]:
        """Return the list of governed tools."""
        return list(self._tools)

    def check_tool_access(self, tool: BaseTool) -> dict[str, Any]:
        """Pre-flight access check for a single tool (fail-open).

        Returns:
            Parsed gateway response dict.
        """
        return enforce_access(
            api_key=self.api_key,
            agent_id=self.agent_id,
            endpoint=_tool_endpoint(tool),
            method="POST",
            fail_closed=False,
            timeout=self.timeout,
            gateway_url=self.gateway_url,
        )

    # -- internal -------------------------------------------------------------

    def _wrap_tool(self, tool: BaseTool) -> BaseTool:
        """Return a copy of *tool* with guarded ``_run`` and ``_arun``."""
        endpoint = _tool_endpoint(tool)
        original_run = tool._run
        original_arun = tool._arun

        api_key = self.api_key
        agent_id = self.agent_id
        fail_closed = self.fail_closed
        timeout = self.timeout
        gateway_url = self.gateway_url

        @functools.wraps(original_run)
        def guarded_run(*args: Any, **kwargs: Any) -> Any:
            result = enforce_access(
                api_key=api_key,
                agent_id=agent_id,
                endpoint=endpoint,
                method="POST",
                fail_closed=fail_closed,
                timeout=timeout,
                gateway_url=gateway_url,
            )
            if result.get("decision") == "deny":
                reason = result.get("reason", "Access denied by AI Identity gateway")
                raise ToolException(reason)
            return original_run(*args, **kwargs)

        @functools.wraps(original_arun)
        async def guarded_arun(*args: Any, **kwargs: Any) -> Any:
            result = await aenforce_access(
                api_key=api_key,
                agent_id=agent_id,
                endpoint=endpoint,
                method="POST",
                fail_closed=fail_closed,
                timeout=timeout,
                gateway_url=gateway_url,
            )
            if result.get("decision") == "deny":
                reason = result.get("reason", "Access denied by AI Identity gateway")
                raise ToolException(reason)
            return await original_arun(*args, **kwargs)

        tool._run = guarded_run  # type: ignore[assignment]
        tool._arun = guarded_arun  # type: ignore[assignment]
        return tool
