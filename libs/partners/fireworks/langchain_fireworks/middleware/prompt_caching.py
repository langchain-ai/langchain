"""Middleware for Fireworks prompt-cache session affinity.

Requires `langchain` for the agent middleware framework. It is imported lazily,
and an `ImportError` with install guidance is raised if it is missing.
`ChatFireworks` is provided by this package (`langchain-fireworks`).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Literal
from warnings import warn

from langchain_fireworks.chat_models import _PROMPT_CACHE_AFFINITY, ChatFireworks

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
    from langgraph.config import get_config
except ImportError as e:
    msg = (
        "FireworksPromptCachingMiddleware requires 'langchain' to be installed "
        f"(missing module: {e.name}). This middleware is designed for use with "
        "LangChain agents. Install it with: pip install langchain"
    )
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)

_UNSUPPORTED_MODEL_BEHAVIORS = ("ignore", "warn", "raise")


def _get_thread_id(request: ModelRequest) -> str | None:
    """Return the thread ID from request runtime metadata or runnable config."""
    # Runtime metadata is available even without async config propagation on
    # Python 3.10. Older runtimes may not expose execution_info.
    execution_info = getattr(request.runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    if isinstance(thread_id, str) and thread_id:
        return thread_id

    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return None


@contextmanager
def _session_affinity(request: ModelRequest) -> Iterator[None]:
    """Scope an affinity default to this call, including any fallback attempts."""
    thread_id = _get_thread_id(request)
    affinity = (
        hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
        if thread_id is not None
        else None
    )
    if affinity is None:
        logger.debug("Fireworks session affinity not applied: no thread_id in config")
    token = _PROMPT_CACHE_AFFINITY.set(affinity)
    try:
        yield
    finally:
        _PROMPT_CACHE_AFFINITY.reset(token)


class FireworksPromptCachingMiddleware(AgentMiddleware):
    """Set Fireworks prompt-cache session affinity from the active thread ID.

    Fireworks prompt caching is enabled by default. This middleware improves
    cache hit rate by pinning session affinity to a SHA-256 hash of
    `config.configurable.thread_id`, so related requests route to the same
    replica and reuse its warm cache. The hexadecimal hash keeps affinity values
    safe for HTTP headers, including when thread IDs contain Unicode.

    The middleware supplies a scoped default that `ChatFireworks` applies to
    `prompt_cache_key` and `extra_headers["x-session-affinity"]` when invoking
    the API. Explicit `user` or `prompt_cache_key` body fields (including
    `extra_body` overrides), or `x-session-affinity` headers on the selected
    model or request take precedence, including on fallback models. No affinity
    is added when no thread ID is configured.

    Generated affinity stays out of shared request settings, so it is never
    forwarded to another provider. This works with either ordering of this
    middleware and `ModelFallbackMiddleware` for a Fireworks primary model.
    """

    def __init__(
        self,
        *,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize the middleware.

        Args:
            unsupported_model_behavior: Behavior when the request model is not
                `ChatFireworks`. `"ignore"` continues without affinity,
                `"warn"` emits a warning, and `"raise"` raises `ValueError`.

        Raises:
            ValueError: If `unsupported_model_behavior` is not one of
                `"ignore"`, `"warn"`, or `"raise"`.
        """
        if unsupported_model_behavior not in _UNSUPPORTED_MODEL_BEHAVIORS:
            msg = (
                "unsupported_model_behavior must be one of "
                f"{_UNSUPPORTED_MODEL_BEHAVIORS}, got {unsupported_model_behavior!r}."
            )
            raise ValueError(msg)
        self.unsupported_model_behavior = unsupported_model_behavior

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Return whether session affinity should be applied to the request.

        Args:
            request: The model request to check.

        Returns:
            `True` if the request model is `ChatFireworks`, else `False`.

        Raises:
            ValueError: If the model is unsupported and
                `unsupported_model_behavior` is `"raise"`.
        """
        if isinstance(request.model, ChatFireworks):
            return True

        msg = (
            "FireworksPromptCachingMiddleware only supports ChatFireworks, "
            f"not {type(request.model).__name__}."
        )
        if self.unsupported_model_behavior == "raise":
            raise ValueError(msg)
        if self.unsupported_model_behavior == "warn":
            warn(msg, stacklevel=3)
        return False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Supply default Fireworks affinity while executing the handler.

        Args:
            request: The outgoing model request.
            handler: Callable that executes the request.

        Returns:
            The result produced by `handler`.
        """
        if not self._should_apply_caching(request):
            return handler(request)
        with _session_affinity(request):
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Supply default Fireworks affinity while awaiting the handler.

        Args:
            request: The outgoing model request.
            handler: Async callable that executes the request.

        Returns:
            The result produced by `handler`.
        """
        if not self._should_apply_caching(request):
            return await handler(request)
        with _session_affinity(request):
            return await handler(request)
