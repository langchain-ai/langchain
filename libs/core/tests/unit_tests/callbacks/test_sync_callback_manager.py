import logging
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain_core.callbacks.manager import handle_event
from langchain_core.messages import HumanMessage


def test_remove_handler() -> None:
    """Test removing handler does not raise an error on removal.

    An handler can be inheritable or not. This test checks that
    removing a handler does not raise an error if the handler
    is not inheritable.
    """
    handler1 = BaseCallbackHandler()
    handler2 = BaseCallbackHandler()
    manager = BaseCallbackManager([handler1], inheritable_handlers=[handler2])
    manager.remove_handler(handler1)
    manager.remove_handler(handler2)


def test_merge_preserves_handler_distinction() -> None:
    """Test that merging managers preserves the distinction between handlers.

    This test verifies the correct behavior of the BaseCallbackManager.merge()
    method. When two managers are merged, their handlers and
    inheritable_handlers should be combined independently.

    Currently, it is expected to xfail until the issue is resolved.
    """
    h1 = BaseCallbackHandler()
    h2 = BaseCallbackHandler()
    ih1 = BaseCallbackHandler()
    ih2 = BaseCallbackHandler()

    m1 = BaseCallbackManager(handlers=[h1], inheritable_handlers=[ih1])
    m2 = BaseCallbackManager(handlers=[h2], inheritable_handlers=[ih2])

    merged = m1.merge(m2)

    assert set(merged.handlers) == {h1, h2}
    assert set(merged.inheritable_handlers) == {ih1, ih2}


def test_handle_event_not_implemented_error_no_args() -> None:
    """No IndexError when NotImplementedError raised with no positional args.

    Regression test for https://github.com/langchain-ai/langchain/issues/31576
    """
    handler = Mock(spec=BaseCallbackHandler)
    handler.on_chat_model_start = Mock(side_effect=NotImplementedError)
    handler.raise_error = False

    # Should not raise IndexError — should log a warning instead
    handle_event(
        [handler],
        "on_chat_model_start",
        None,
    )


def test_handle_event_not_implemented_error_with_args() -> None:
    """Fallback to on_llm_start when on_chat_model_start raises NotImplementedError."""
    llm_start_calls: list[Any] = []

    class FallbackHandler(BaseCallbackHandler):
        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            **kwargs: Any,  # noqa: ARG002
        ) -> None:
            llm_start_calls.append((serialized, prompts))

    handler = FallbackHandler()

    serialized = {"name": "test_model"}
    messages = [[HumanMessage(content="hello")]]

    handle_event(
        [handler],
        "on_chat_model_start",
        None,
        serialized,
        messages,
        run_id=uuid4(),
    )

    assert len(llm_start_calls) == 1
    assert llm_start_calls[0][0] == serialized
    # The fallback should have converted messages to strings
    assert isinstance(llm_start_calls[0][1], list)
    assert isinstance(llm_start_calls[0][1][0], str)


def test_handle_event_non_chat_model_not_implemented(caplog: Any) -> None:
    """handle_event logs a warning for NotImplementedError on non-chat events."""
    handler = Mock(spec=BaseCallbackHandler)
    handler.on_llm_start = Mock(side_effect=NotImplementedError("not implemented"))
    handler.raise_error = False
    handler.__class__.__name__ = "MockHandler"

    with caplog.at_level(logging.WARNING):
        handle_event(
            [handler],
            "on_llm_start",
            None,
        )

    assert "NotImplementedError" in caplog.text
