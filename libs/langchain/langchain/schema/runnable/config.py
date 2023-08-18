from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManager, AsyncCallbackManager


class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

    tags: List[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    metadata: Dict[str, Any]
    """
    Metadata for this call and any sub-calls (eg. a Chain calling an LLM).
    Keys should be strings, values should be JSON-serializable.
    """

    callbacks: Callbacks
    """
    Callbacks for this call and any sub-calls (eg. a Chain calling an LLM).
    Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
    """

    _locals: Dict[str, Any]
    """
    Local variables
    """


def ensure_config(config: Optional[RunnableConfig]) -> RunnableConfig:
    empty = RunnableConfig(tags=[], metadata={}, callbacks=None, _locals={})
    if config is not None:
        empty.update(config)
    return empty


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )
