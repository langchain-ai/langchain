"""Governance middleware for LangChain agents.

Implements per-agent policy enforcement and tamper-evident audit logging as
middleware that can be added to any ``langchain_v1`` agent without swapping
out the LLM or wrapping individual tools manually.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

from langchain_ai_identity._gateway import (
    _DEFAULT_TIMEOUT,
    _LLM_ENDPOINT,
    aenforce_access,
    apost_audit,
    enforce_access,
    post_audit,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)


class AIIdentityGovernanceMiddleware:
    """LangChain middleware that enforces AI Identity governance on agent execution.

    Add this middleware to any langchain_v1 agent to get per-agent policy
    enforcement and tamper-evident audit logging without changing your LLM
    provider or wrapping individual tools.

    Example::

        from langchain_ai_identity.middleware import AIIdentityGovernanceMiddleware

        middleware = AIIdentityGovernanceMiddleware(
            agent_id="<your-agent-uuid>",
            api_key="aid_sk_...",
        )
        # Add to agent via middleware=[middleware] in agent config

    Args:
        agent_id: UUID of the registered AI Identity agent.
        api_key: The ``aid_sk_`` prefixed runtime key.
        fail_closed: When ``True`` (default), denials raise; ``False`` logs warning.
        timeout: HTTP timeout in seconds for gateway calls.
        gateway_url: Override for the gateway base URL.
        api_url: Override for the audit API base URL.
        audit_enabled: Whether to post audit entries (default ``True``).
        enforce_on_model: Whether to enforce policy on model calls (default ``True``).
        enforce_on_tools: Whether to enforce policy on tool calls (default ``True``).
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        *,
        fail_closed: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
        gateway_url: str | None = None,
        api_url: str | None = None,
        audit_enabled: bool = True,
        enforce_on_model: bool = True,
        enforce_on_tools: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.api_key = api_key
        self.fail_closed = fail_closed
        self.timeout = timeout
        self.gateway_url = gateway_url
        self.api_url = api_url
        self.audit_enabled = audit_enabled
        self.enforce_on_model = enforce_on_model
        self.enforce_on_tools = enforce_on_tools

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the middleware name."""
        return "AIIdentityGovernanceMiddleware"

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def validate_config(self) -> None:
        """Validate that required configuration is present.

        Raises:
            ValueError: If ``agent_id`` or ``api_key`` is empty.
        """
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if not self.api_key:
            raise ValueError("api_key must not be empty")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_decision(self, result: dict[str, Any], endpoint: str) -> str:
        """Inspect an enforce_access result and raise/warn on denial.

        Args:
            result: Parsed JSON from the gateway.
            endpoint: The endpoint that was checked (for error messages).

        Returns:
            The decision string (``"allow"`` or ``"deny"``).

        Raises:
            PermissionError: If denied and ``fail_closed`` is ``True``.
        """
        decision = result.get("decision", "deny")
        if decision != "allow":
            reason = result.get("reason", "no reason provided")
            msg = (
                f"[AI Identity] Access denied for agent {self.agent_id!r} "
                f"on {endpoint!r}: {reason}"
            )
            if self.fail_closed:
                raise PermissionError(msg)
            warnings.warn(msg, stacklevel=3)
        return decision

    # ------------------------------------------------------------------
    # Model call enforcement
    # ------------------------------------------------------------------

    def enforce_model_call(self, call_fn: Callable, **kwargs: Any) -> Any:
        """Enforce governance policy around a synchronous model call.

        Args:
            call_fn: The callable that performs the actual LLM invocation.
            **kwargs: Keyword arguments forwarded to *call_fn*.

        Returns:
            The result of *call_fn*.

        Raises:
            PermissionError: If the policy denies the call and
                ``fail_closed`` is ``True``.
        """
        decision = "allow"

        if self.enforce_on_model:
            result = enforce_access(
                api_key=self.api_key,
                agent_id=self.agent_id,
                endpoint=_LLM_ENDPOINT,
                fail_closed=self.fail_closed,
                timeout=self.timeout,
                gateway_url=self.gateway_url,
            )
            decision = self._check_decision(result, _LLM_ENDPOINT)

        start = time.perf_counter()
        response = call_fn(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if self.audit_enabled:
            post_audit(
                api_key=self.api_key,
                agent_id=self.agent_id,
                event_type="llm_call",
                endpoint=_LLM_ENDPOINT,
                decision=decision,
                latency_ms=latency_ms,
                fail_closed=False,
                timeout=self.timeout,
                api_url=self.api_url,
            )

        return response

    async def aenforce_model_call(self, call_fn: Callable, **kwargs: Any) -> Any:
        """Enforce governance policy around an asynchronous model call.

        Args:
            call_fn: The async callable that performs the actual LLM invocation.
            **kwargs: Keyword arguments forwarded to *call_fn*.

        Returns:
            The result of *call_fn*.

        Raises:
            PermissionError: If the policy denies the call and
                ``fail_closed`` is ``True``.
        """
        decision = "allow"

        if self.enforce_on_model:
            result = await aenforce_access(
                api_key=self.api_key,
                agent_id=self.agent_id,
                endpoint=_LLM_ENDPOINT,
                fail_closed=self.fail_closed,
                timeout=self.timeout,
                gateway_url=self.gateway_url,
            )
            decision = self._check_decision(result, _LLM_ENDPOINT)

        start = time.perf_counter()
        response = await call_fn(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if self.audit_enabled:
            await apost_audit(
                api_key=self.api_key,
                agent_id=self.agent_id,
                event_type="llm_call",
                endpoint=_LLM_ENDPOINT,
                decision=decision,
                latency_ms=latency_ms,
                fail_closed=False,
                timeout=self.timeout,
                api_url=self.api_url,
            )

        return response

    # ------------------------------------------------------------------
    # Tool call enforcement
    # ------------------------------------------------------------------

    def enforce_tool_call(
        self, tool_name: str, call_fn: Callable, **kwargs: Any
    ) -> Any:
        """Enforce governance policy around a synchronous tool call.

        Args:
            tool_name: The name of the tool being invoked.
            call_fn: The callable that performs the actual tool invocation.
            **kwargs: Keyword arguments forwarded to *call_fn*.

        Returns:
            The result of *call_fn*.

        Raises:
            PermissionError: If the policy denies the call and
                ``fail_closed`` is ``True``.
        """
        endpoint = f"/tools/{tool_name}"
        decision = "allow"

        if self.enforce_on_tools:
            result = enforce_access(
                api_key=self.api_key,
                agent_id=self.agent_id,
                endpoint=endpoint,
                fail_closed=self.fail_closed,
                timeout=self.timeout,
                gateway_url=self.gateway_url,
            )
            decision = self._check_decision(result, endpoint)

        start = time.perf_counter()
        response = call_fn(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if self.audit_enabled:
            post_audit(
                api_key=self.api_key,
                agent_id=self.agent_id,
                event_type="tool_call",
                endpoint=endpoint,
                decision=decision,
                latency_ms=latency_ms,
                fail_closed=False,
                timeout=self.timeout,
                api_url=self.api_url,
            )

        return response

    async def aenforce_tool_call(
        self, tool_name: str, call_fn: Callable, **kwargs: Any
    ) -> Any:
        """Enforce governance policy around an asynchronous tool call.

        Args:
            tool_name: The name of the tool being invoked.
            call_fn: The async callable that performs the actual tool invocation.
            **kwargs: Keyword arguments forwarded to *call_fn*.

        Returns:
            The result of *call_fn*.

        Raises:
            PermissionError: If the policy denies the call and
                ``fail_closed`` is ``True``.
        """
        endpoint = f"/tools/{tool_name}"
        decision = "allow"

        if self.enforce_on_tools:
            result = await aenforce_access(
                api_key=self.api_key,
                agent_id=self.agent_id,
                endpoint=endpoint,
                fail_closed=self.fail_closed,
                timeout=self.timeout,
                gateway_url=self.gateway_url,
            )
            decision = self._check_decision(result, endpoint)

        start = time.perf_counter()
        response = await call_fn(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if self.audit_enabled:
            await apost_audit(
                api_key=self.api_key,
                agent_id=self.agent_id,
                event_type="tool_call",
                endpoint=endpoint,
                decision=decision,
                latency_ms=latency_ms,
                fail_closed=False,
                timeout=self.timeout,
                api_url=self.api_url,
            )

        return response
