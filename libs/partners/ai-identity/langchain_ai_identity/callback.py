"""AI Identity callback handlers for LangChain audit logging.

Provides synchronous and asynchronous callback handlers that post audit
events to the AI Identity API whenever LLM or tool invocations occur.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_ai_identity._gateway import (
    _DEFAULT_TIMEOUT,
    _LLM_ENDPOINT,
    apost_audit,
    post_audit,
)

logger = logging.getLogger(__name__)


class AIIdentityCallbackHandler(BaseCallbackHandler):
    """Synchronous callback handler that logs audit events to AI Identity.

    Tracks LLM and tool lifecycle events (start, end, error) and posts
    structured audit entries via the AI Identity management API.

    Args:
        api_key: AI Identity API key.
        agent_id: Unique identifier for the agent.
        fail_closed: If ``True``, raise on audit failures. Defaults to ``True``.
        timeout: HTTP timeout in seconds. Defaults to ``5.0``.
        api_url: Override for the AI Identity API base URL.
    """

    raise_error = False

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        *,
        fail_closed: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
        api_url: str | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.agent_id = agent_id
        self.fail_closed = fail_closed
        self.timeout = timeout
        self.api_url = api_url
        self._llm_start_times: dict[UUID, float] = {}

    # -- LLM events ----------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record the start time of an LLM invocation."""
        self._llm_start_times[run_id] = time.monotonic()
        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="llm_start",
            endpoint=_LLM_ENDPOINT,
            decision="allow",
            metadata={"model": serialized.get("name", "unknown")},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the completion of an LLM invocation with latency."""
        latency_ms: float | None = None
        start = self._llm_start_times.pop(run_id, None)
        if start is not None:
            latency_ms = (time.monotonic() - start) * 1000

        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="llm_end",
            endpoint=_LLM_ENDPOINT,
            decision="allow",
            latency_ms=latency_ms,
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    # -- Tool events ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the start of a tool invocation."""
        tool_name = serialized.get("name", "unknown")
        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_start",
            endpoint=f"/tools/{tool_name}",
            decision="allow",
            metadata={"tool": tool_name},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the completion of a tool invocation."""
        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_end",
            endpoint="/tools/unknown",
            decision="allow",
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    # -- Error events ---------------------------------------------------------

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log a chain-level error."""
        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="chain_error",
            endpoint=_LLM_ENDPOINT,
            decision="deny",
            metadata={"error": str(error)},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log a tool-level error."""
        post_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_error",
            endpoint="/tools/unknown",
            decision="deny",
            metadata={"error": str(error)},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )


class AIIdentityAsyncCallbackHandler(AsyncCallbackHandler):
    """Asynchronous callback handler that logs audit events to AI Identity.

    Async counterpart of :class:`AIIdentityCallbackHandler`.  Uses
    ``apost_audit`` for non-blocking HTTP calls.

    Args:
        api_key: AI Identity API key.
        agent_id: Unique identifier for the agent.
        fail_closed: If ``True``, raise on audit failures. Defaults to ``True``.
        timeout: HTTP timeout in seconds. Defaults to ``5.0``.
        api_url: Override for the AI Identity API base URL.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        *,
        fail_closed: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
        api_url: str | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.agent_id = agent_id
        self.fail_closed = fail_closed
        self.timeout = timeout
        self.api_url = api_url
        self._llm_start_times: dict[UUID, float] = {}

    # -- LLM events ----------------------------------------------------------

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Record the start time of an LLM invocation."""
        self._llm_start_times[run_id] = time.monotonic()
        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="llm_start",
            endpoint=_LLM_ENDPOINT,
            decision="allow",
            metadata={"model": serialized.get("name", "unknown")},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the completion of an LLM invocation with latency."""
        latency_ms: float | None = None
        start = self._llm_start_times.pop(run_id, None)
        if start is not None:
            latency_ms = (time.monotonic() - start) * 1000

        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="llm_end",
            endpoint=_LLM_ENDPOINT,
            decision="allow",
            latency_ms=latency_ms,
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    # -- Tool events ----------------------------------------------------------

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the start of a tool invocation."""
        tool_name = serialized.get("name", "unknown")
        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_start",
            endpoint=f"/tools/{tool_name}",
            decision="allow",
            metadata={"tool": tool_name},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log the completion of a tool invocation."""
        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_end",
            endpoint="/tools/unknown",
            decision="allow",
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    # -- Error events ---------------------------------------------------------

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log a chain-level error."""
        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="chain_error",
            endpoint=_LLM_ENDPOINT,
            decision="deny",
            metadata={"error": str(error)},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Log a tool-level error."""
        await apost_audit(
            api_key=self.api_key,
            agent_id=self.agent_id,
            event_type="tool_error",
            endpoint="/tools/unknown",
            decision="deny",
            metadata={"error": str(error)},
            fail_closed=self.fail_closed,
            timeout=self.timeout,
            api_url=self.api_url,
        )
