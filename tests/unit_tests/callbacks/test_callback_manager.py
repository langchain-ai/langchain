"""Test CallbackManager."""

from langchain.callbacks.base import BaseCallbackManager, CallbackManager
from langchain.callbacks.shared import SharedCallbackManager
from langchain.schema import LLMResult
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def _test_callback_manager(
    manager: BaseCallbackManager, *handlers: FakeCallbackHandler
) -> None:
    """Test the CallbackManager."""
    manager.on_llm_start({}, [])
    manager.on_llm_end(LLMResult(generations=[]))
    manager.on_llm_error(Exception())
    manager.on_chain_start({}, {})
    manager.on_chain_end({})
    manager.on_chain_error(Exception())
    manager.on_tool_start({}, "", "")
    manager.on_tool_end("")
    manager.on_tool_error(Exception())
    for handler in handlers:
        assert handler.starts == 3
        assert handler.ends == 3
        assert handler.errors == 3


def test_callback_manager() -> None:
    """Test the CallbackManager."""
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()
    manager = CallbackManager([handler1, handler2])
    _test_callback_manager(manager, handler1, handler2)


def test_shared_callback_manager() -> None:
    """Test the SharedCallbackManager."""
    manager1 = SharedCallbackManager()
    manager2 = SharedCallbackManager()

    assert manager1 is manager2

    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()
    manager1.add_handler(handler1)
    manager2.add_handler(handler2)
    _test_callback_manager(manager1, handler1, handler2)
