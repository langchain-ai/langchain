"""Tests for handle_event and _ahandle_event_for_handler fallback behavior.

Covers the NotImplementedError fallback from on_chat_model_start to on_llm_start.
Handlers must declare `serialized` and `messages` as explicit positional args
(not *args) — see on_chat_model_start docstring for details.

See: https://github.com/langchain-ai/langchain/issues/31576
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import (
    _ahandle_event_for_handler,
    handle_event,
)
from langchain_core.messages import BaseMessage, HumanMessage


class _FallbackChatHandler(BaseCallbackHandler):
    """Handler that correctly declares the required args but raises NotImplementedError.

    This triggers the fallback to on_llm_start, as documented.
    """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass


class _FallbackChatHandlerAsync(BaseCallbackHandler):
    """Async-compatible handler; raises NotImplementedError for on_chat_model_start."""

    run_inline = True

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_handle_event_chat_model_start_fallback_to_llm_start() -> None:
    """on_chat_model_start raises NotImplementedError → falls back to on_llm_start."""
    handler = _FallbackChatHandler()
    handler.on_llm_start = MagicMock()  # type: ignore[method-assign]

    serialized = {"name": "test"}
    messages = [[HumanMessage(content="hello")]]

    handle_event(
        [handler],
        "on_chat_model_start",
        "ignore_chat_model",
        serialized,
        messages,
    )

    handler.on_llm_start.assert_called_once()


def test_handle_event_other_event_not_implemented_logs_warning() -> None:
    """Non-chat_model_start events that raise NotImplementedError log a warning."""

    class _Handler(BaseCallbackHandler):
        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    handler = _Handler()

    # Should not raise — logs a warning instead
    handle_event(
        [handler],
        "on_llm_start",
        "ignore_llm",
        {"name": "test"},
        ["prompt"],
    )


@pytest.mark.asyncio
async def test_ahandle_event_chat_model_start_fallback_to_llm_start() -> None:
    """Async: on_chat_model_start NotImplementedError falls back to on_llm_start."""
    handler = _FallbackChatHandlerAsync()
    handler.on_llm_start = MagicMock()  # type: ignore[method-assign]

    serialized = {"name": "test"}
    messages = [[HumanMessage(content="hello")]]

    await _ahandle_event_for_handler(
        handler,
        "on_chat_model_start",
        "ignore_chat_model",
        serialized,
        messages,
    )

    handler.on_llm_start.assert_called_once()


@pytest.mark.asyncio
async def test_ahandle_event_other_event_not_implemented_logs_warning() -> None:
    """Async: non-chat_model_start events log warning on NotImplementedError."""

    class _Handler(BaseCallbackHandler):
        run_inline = True

        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    handler = _Handler()

    await _ahandle_event_for_handler(
        handler,
        "on_llm_start",
        "ignore_llm",
        {"name": "test"},
        ["prompt"],
    )
