"""Tests for handle_event and _ahandle_event_for_handler fallback behavior.

Specifically covers the NotImplementedError fallback from on_chat_model_start
to on_llm_start, ensuring no IndexError when args are insufficient.

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
from langchain_core.messages import HumanMessage


class _NotImplementedChatHandler(BaseCallbackHandler):
    """Handler that raises NotImplementedError for on_chat_model_start."""

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NotImplementedChatHandlerAsync(BaseCallbackHandler):
    """Async-compatible handler that raises NotImplementedError."""

    run_inline = True

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_handle_event_chat_model_start_fallback_with_args() -> None:
    """on_chat_model_start falls back to on_llm_start when args are provided."""
    handler = _NotImplementedChatHandler()
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


def test_handle_event_chat_model_start_no_args_no_crash() -> None:
    """on_chat_model_start with no positional args should not raise IndexError."""
    handler = _NotImplementedChatHandler()

    # Before the fix, this would raise IndexError: tuple index out of range
    handle_event(
        [handler],
        "on_chat_model_start",
        "ignore_chat_model",
    )


def test_handle_event_chat_model_start_one_arg_no_crash() -> None:
    """on_chat_model_start with only one positional arg should not raise IndexError."""
    handler = _NotImplementedChatHandler()

    handle_event(
        [handler],
        "on_chat_model_start",
        "ignore_chat_model",
        {"name": "test"},
    )


def test_handle_event_other_event_not_implemented() -> None:
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
async def test_ahandle_event_chat_model_start_fallback_with_args() -> None:
    """Async: on_chat_model_start falls back to on_llm_start with proper args."""
    handler = _NotImplementedChatHandlerAsync()
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
async def test_ahandle_event_chat_model_start_no_args_no_crash() -> None:
    """Async: on_chat_model_start with no args should not raise IndexError."""
    handler = _NotImplementedChatHandlerAsync()

    # Before the fix, this would raise IndexError: tuple index out of range
    await _ahandle_event_for_handler(
        handler,
        "on_chat_model_start",
        "ignore_chat_model",
    )


@pytest.mark.asyncio
async def test_ahandle_event_chat_model_start_one_arg_no_crash() -> None:
    """Async: on_chat_model_start with one arg should not raise IndexError."""
    handler = _NotImplementedChatHandlerAsync()

    await _ahandle_event_for_handler(
        handler,
        "on_chat_model_start",
        "ignore_chat_model",
        {"name": "test"},
    )


@pytest.mark.asyncio
async def test_ahandle_event_other_event_not_implemented() -> None:
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
