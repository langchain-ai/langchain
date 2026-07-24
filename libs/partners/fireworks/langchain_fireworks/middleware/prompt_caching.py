"""Middleware for Fireworks prompt-cache session affinity.

Requires `langchain` for the agent middleware framework. It is imported lazily,
and an `ImportError` with install guidance is raised if it is missing.
`ChatFireworks` is provided by this package (`langchain-fireworks`).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal
from warnings import warn

from langchain_fireworks.chat_models import ChatFireworks

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

_SESSION_AFFINITY_HEADER = "x-session-affinity"
_USER_MANAGED_SETTINGS = ("user", "prompt_cache_key")
_UNSUPPORTED_MODEL_BEHAVIORS = ("ignore", "warn", "raise")


def _get_thread_id() -> str | None:
    """Return a non-empty `config.configurable.thread_id`, if present."""
    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = (config.get("configurable") or {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return None


def _has_session_affinity_header(headers: Mapping[Any, Any]) -> bool:
    """Return whether headers already contain `x-session-affinity`."""
    return any(
        isinstance(key, str) and key.lower() == _SESSION_AFFINITY_HEADER
        for key in headers
    )


def _get_effective_model_settings(request: ModelRequest) -> dict[str, Any]:
    """Combine model defaults with settings supplied for this request."""
    raw_model_settings = getattr(request.model, "model_kwargs", None)
    model_settings = (
        dict(raw_model_settings) if isinstance(raw_model_settings, Mapping) else {}
    )

    model_headers = model_settings.get("extra_headers")
    request_headers = request.model_settings.get("extra_headers")
    model_settings.update(request.model_settings)
    if isinstance(model_headers, Mapping) and isinstance(request_headers, Mapping):
        model_settings["extra_headers"] = {**model_headers, **request_headers}
    return model_settings


class FireworksPromptCachingMiddleware(AgentMiddleware):
    """Set Fireworks prompt-cache session affinity from the active thread ID.

    Fireworks prompt caching is enabled by default. This middleware improves
    cache hit rate by pinning session affinity to
    `config.configurable.thread_id`, so related requests route to the same
    replica and reuse its warm cache.

    The middleware injects both `prompt_cache_key` and
    `extra_headers["x-session-affinity"]`. It leaves requests unchanged when no
    thread ID is configured or the caller already supplies `user`,
    `prompt_cache_key`, or an `x-session-affinity` header.
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

    def _apply_session_affinity(self, request: ModelRequest) -> ModelRequest | None:
        """Return a request with session affinity applied, or `None` to no-op."""
        thread_id = _get_thread_id()
        if thread_id is None:
            logger.debug(
                "Fireworks session affinity not applied: "
                "no thread_id in runnable config"
            )
            return None

        model_settings = _get_effective_model_settings(request)
        if any(model_settings.get(key) for key in _USER_MANAGED_SETTINGS):
            return None

        raw_headers = model_settings.get("extra_headers")
        if raw_headers is None:
            headers: dict[Any, Any] = {}
        elif isinstance(raw_headers, Mapping):
            if _has_session_affinity_header(raw_headers):
                return None
            headers = dict(raw_headers)
        else:
            logger.warning(
                "Cannot set Fireworks session affinity because extra_headers is %s",
                type(raw_headers).__name__,
            )
            return None

        headers[_SESSION_AFFINITY_HEADER] = thread_id
        # Pin affinity on both channels: the typed `prompt_cache_key` field
        # (preferred by newer Fireworks endpoints) and the `x-session-affinity`
        # header (honored by endpoints that only read the raw header). Writing the
        # same value to both is safe. Only this request's settings are carried
        # forward -- the model's own `model_kwargs` already reach the API via the
        # model itself, so re-binding them here is unnecessary.
        new_settings = {
            **request.model_settings,
            "prompt_cache_key": thread_id,
            "extra_headers": headers,
        }
        logger.debug("Set Fireworks prompt-cache session affinity")
        return request.override(model_settings=new_settings)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Inject Fireworks session affinity before delegating to the handler.

        Args:
            request: The outgoing model request.
            handler: Callable that executes the (possibly modified) request.

        Returns:
            The result produced by `handler`.
        """
        if not self._should_apply_caching(request):
            return handler(request)
        new_request = self._apply_session_affinity(request)
        return handler(request if new_request is None else new_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Inject Fireworks session affinity before delegating asynchronously.

        Args:
            request: The outgoing model request.
            handler: Async callable that executes the (possibly modified) request.

        Returns:
            The result produced by `handler`.
        """
        if not self._should_apply_caching(request):
            return await handler(request)
        new_request = self._apply_session_affinity(request)
        return await handler(request if new_request is None else new_request)
