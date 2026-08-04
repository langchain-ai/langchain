"""Config helper for middleware-internal model calls.

Prevents internal model calls from inheriting parent streaming callbacks and
leaking tokens into the user-facing stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.runnables.config import ensure_config
from langchain_core.tracers._streaming import (
    _StreamingCallbackHandler,
    _V2StreamingCallbackHandler,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig


def _inherited_handlers(callbacks: Any) -> list[BaseCallbackHandler]:
    """Flatten the current callbacks (list or manager) into a deduplicated list."""
    if isinstance(callbacks, list):
        return callbacks
    if isinstance(callbacks, BaseCallbackManager):
        seen: set[int] = set()
        handlers: list[BaseCallbackHandler] = []
        for handler in (*callbacks.handlers, *callbacks.inheritable_handlers):
            if id(handler) not in seen:
                seen.add(id(handler))
                handlers.append(handler)
        return handlers
    return []


def internal_call_config(lc_source: str) -> RunnableConfig:
    """Build config for a middleware-internal, non-user-visible model call.

    Preserves tracing and non-streaming callbacks while removing handlers that
    would stream tokens to the user.

    Args:
        lc_source: Source recorded in `metadata["lc_source"]`.

    Returns:
        Config with streaming callback handlers removed.
    """
    config = ensure_config()
    handlers = _inherited_handlers(config.get("callbacks"))
    filtered_handlers = [
        handler
        for handler in handlers
        if not isinstance(handler, (_StreamingCallbackHandler, _V2StreamingCallbackHandler))
    ]

    return {
        "callbacks": filtered_handlers,
        "metadata": {**config.get("metadata", {}), "lc_source": lc_source},
    }
