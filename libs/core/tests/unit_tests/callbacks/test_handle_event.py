"""Tests for handle_event in callback manager."""

from unittest.mock import Mock

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import handle_event


class _RaisingHandler(BaseCallbackHandler):
    """Handler that raises NotImplementedError on on_chat_model_start."""

    def on_chat_model_start(self, *args, **kwargs):
        raise NotImplementedError("not implemented")


def test_handle_event_not_implemented_error_no_args():
    """Test that handle_event does not raise IndexError when a handler raises
    NotImplementedError for on_chat_model_start and no positional args are given.

    Regression test for https://github.com/langchain-ai/langchain/issues/31576
    """
    handler = _RaisingHandler()
    # Should not raise IndexError — should log a warning instead
    handle_event(
        [handler],
        "on_chat_model_start",
        ignore_condition_name=None,
    )


def test_handle_event_generic_exception_is_logged():
    """Test that generic exceptions in handlers are caught and logged."""
    handler = Mock(spec=BaseCallbackHandler)
    handler.on_llm_start = Mock(side_effect=KeyError("test"))
    handler.raise_error = False
    handler.ignore_llm = False

    # Should not raise — the KeyError should be caught and logged
    handle_event(
        [handler],
        "on_llm_start",
        "ignore_llm",
    )
