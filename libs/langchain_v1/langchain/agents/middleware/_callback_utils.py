"""Callback utilities for middleware internal model calls."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tracers.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks.base import BaseCallbackHandler


@lru_cache(maxsize=1)
def _get_usage_handler_type() -> type[BaseCallbackHandler] | None:
    """Get UsageMetadataCallbackHandler type if available."""
    try:
        from langchain_core.callbacks.usage import UsageMetadataCallbackHandler  # noqa: PLC0415
    except ImportError:
        return None
    else:
        return UsageMetadataCallbackHandler


def get_internal_call_config(
    *,
    additional_callback_types: Sequence[type[BaseCallbackHandler]] | None = None,
) -> RunnableConfig:
    """Get a RunnableConfig appropriate for internal middleware model calls.

    Internal calls (summarization, tool selection, tool emulation) need:
    - Tracing callbacks (LangSmith): PRESERVED for observability
    - Usage callbacks: PRESERVED for token counting
    - Streaming callbacks: BLOCKED to prevent token leakage

    Args:
        additional_callback_types: Optional callback types to preserve beyond
            the default BaseTracer and UsageMetadataCallbackHandler.

    Returns:
        RunnableConfig with filtered callbacks suitable for internal calls.
    """
    current_config = ensure_config()
    callbacks = current_config.get("callbacks")

    if callbacks is None:
        return {"callbacks": []}

    # Extract handlers from CallbackManager or use list directly
    handlers: list[BaseCallbackHandler]
    if hasattr(callbacks, "handlers"):
        handlers = callbacks.handlers
    elif isinstance(callbacks, list):
        handlers = callbacks
    else:
        return {"callbacks": []}  # type: ignore[unreachable]

    # Build whitelist of callback types that SHOULD see internal calls
    allowed_types: tuple[type[BaseCallbackHandler], ...] = (BaseTracer,)

    usage_handler_type = _get_usage_handler_type()
    if usage_handler_type is not None:
        allowed_types = (*allowed_types, usage_handler_type)

    if additional_callback_types:
        allowed_types = (*allowed_types, *additional_callback_types)

    filtered_callbacks = [handler for handler in handlers if isinstance(handler, allowed_types)]

    return {"callbacks": filtered_callbacks}
