"""Test CallbackManager."""

from typing import Any, Dict, List

from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManager,
)
from langchain.callbacks.shared import SharedCallbackManager
from langchain.schema import LLMResult


class FakeCallbackHandler(BaseCallbackHandler):
    """Fake callback handler for testing."""

    def __init__(self) -> None:
        """Initialize the mock callback handler."""
        self.starts = 0
        self.ends = 0
        self.errors = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Run when LLM starts running."""
        self.starts += 1

    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """Run when LLM ends running."""
        self.ends += 1

    def on_llm_error(self, error: Exception) -> None:
        """Run when LLM errors."""
        self.errors += 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Run when chain starts running."""
        self.starts += 1

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Run when chain ends running."""
        self.ends += 1

    def on_chain_error(self, error: Exception) -> None:
        """Run when chain errors."""
        self.errors += 1

    def on_tool_start(
        self, serialized: Dict[str, Any], action: str, tool_input: str, **extra: str
    ) -> None:
        """Run when tool starts running."""
        self.starts += 1

    def on_tool_end(self, output: str) -> None:
        """Run when tool ends running."""
        self.ends += 1

    def on_tool_error(self, error: Exception) -> None:
        """Run when tool errors."""
        self.errors += 1


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
