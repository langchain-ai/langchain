"""AI Identity-governed wrapper around ChatOpenAI.

Intercepts every LLM call with a gateway policy check before forwarding
to the upstream OpenAI model.  Automatically injects the
:class:`~langchain_ai_identity.callback.AIIdentityCallbackHandler` for
audit logging.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from langchain_ai_identity._gateway import (
    _DEFAULT_TIMEOUT,
    _LLM_ENDPOINT,
    aenforce_access,
    enforce_access,
)
from langchain_ai_identity.callback import AIIdentityCallbackHandler

logger = logging.getLogger(__name__)


class AIIdentityChatOpenAI(ChatOpenAI):  # type: ignore[misc]
    """ChatOpenAI subclass with AI Identity gateway enforcement.

    Every call to ``_generate``, ``_agenerate``, ``_stream``, or
    ``_astream`` is preceded by a gateway policy check.  If the policy
    denies the request and ``fail_closed`` is ``True``, a
    :class:`PermissionError` is raised; otherwise a warning is emitted
    and the call proceeds.

    Args:
        agent_id: Unique identifier for the agent.
        ai_identity_api_key: API key for the AI Identity platform.
        fail_closed: Block on deny when ``True`` (default).
        ai_identity_timeout: HTTP timeout for gateway calls in seconds.
        gateway_url: Override for the AI Identity gateway base URL.
    """

    agent_id: str
    ai_identity_api_key: str
    fail_closed: bool = True
    ai_identity_timeout: float = _DEFAULT_TIMEOUT
    gateway_url: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._inject_callback_handler()

    # -- callback injection ---------------------------------------------------

    def _inject_callback_handler(self) -> None:
        """Add the AI Identity callback handler if not already present."""
        if self.callbacks is None:
            self.callbacks = []
        for cb in self.callbacks:  # type: ignore[union-attr]
            if isinstance(cb, AIIdentityCallbackHandler):
                return
        self.callbacks.append(  # type: ignore[union-attr]
            AIIdentityCallbackHandler(
                api_key=self.ai_identity_api_key,
                agent_id=self.agent_id,
                fail_closed=self.fail_closed,
                timeout=self.ai_identity_timeout,
            )
        )

    # -- policy helpers -------------------------------------------------------

    def _check_access(self) -> dict[str, Any]:
        """Synchronous gateway policy check."""
        result = enforce_access(
            api_key=self.ai_identity_api_key,
            agent_id=self.agent_id,
            endpoint=_LLM_ENDPOINT,
            method="POST",
            fail_closed=self.fail_closed,
            timeout=self.ai_identity_timeout,
            gateway_url=self.gateway_url,
        )
        if result.get("decision") == "deny":
            reason = result.get("reason", "Access denied by AI Identity gateway")
            if self.fail_closed:
                raise PermissionError(reason)
            warnings.warn(reason, stacklevel=3)
        return result

    async def _acheck_access(self) -> dict[str, Any]:
        """Asynchronous gateway policy check."""
        result = await aenforce_access(
            api_key=self.ai_identity_api_key,
            agent_id=self.agent_id,
            endpoint=_LLM_ENDPOINT,
            method="POST",
            fail_closed=self.fail_closed,
            timeout=self.ai_identity_timeout,
            gateway_url=self.gateway_url,
        )
        if result.get("decision") == "deny":
            reason = result.get("reason", "Access denied by AI Identity gateway")
            if self.fail_closed:
                raise PermissionError(reason)
            warnings.warn(reason, stacklevel=3)
        return result

    # -- overrides ------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Enforce access then delegate to ChatOpenAI._generate."""
        self._check_access()
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Enforce access then delegate to ChatOpenAI._agenerate."""
        await self._acheck_access()
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Enforce access then delegate to ChatOpenAI._stream."""
        self._check_access()
        yield from super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Enforce access then delegate to ChatOpenAI._astream."""
        await self._acheck_access()
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk
