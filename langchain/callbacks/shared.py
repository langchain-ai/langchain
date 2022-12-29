"""A shared CallbackManager."""

import threading
from typing import Any, Dict, List

from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManager,
)
from langchain.schema import AgentAction, LLMResult


class Singleton:
    """A thread-safe singleton class that can be inherited from."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> Any:
        """Create a new shared instance of the class."""
        if cls._instance is None:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance


class SharedCallbackManager(Singleton, BaseCallbackManager):
    """A thread-safe singleton CallbackManager."""

    _callback_manager: CallbackManager = CallbackManager([])

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Run when LLM starts running."""
        with self._lock:
            self._callback_manager.on_llm_start(serialized, prompts, **extra)

    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """Run when LLM ends running."""
        with self._lock:
            self._callback_manager.on_llm_end(response)

    def on_llm_error(self, error: Exception) -> None:
        """Run when LLM errors."""
        with self._lock:
            self._callback_manager.on_llm_error(error)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Run when chain starts running."""
        with self._lock:
            self._callback_manager.on_chain_start(serialized, inputs, **extra)

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Run when chain ends running."""
        with self._lock:
            self._callback_manager.on_chain_end(outputs)

    def on_chain_error(self, error: Exception) -> None:
        """Run when chain errors."""
        with self._lock:
            self._callback_manager.on_chain_error(error)

    def on_tool_start(
        self, serialized: Dict[str, Any], action: AgentAction, **extra: str
    ) -> None:
        """Run when tool starts running."""
        with self._lock:
            self._callback_manager.on_tool_start(serialized, action, **extra)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        with self._lock:
            self._callback_manager.on_tool_end(output, **kwargs)

    def on_tool_error(self, error: Exception) -> None:
        """Run when tool errors."""
        with self._lock:
            self._callback_manager.on_tool_error(error)

    def add_handler(self, callback: BaseCallbackHandler) -> None:
        """Add a callback to the callback manager."""
        with self._lock:
            self._callback_manager.add_handler(callback)

    def remove_handler(self, callback: BaseCallbackHandler) -> None:
        """Remove a callback from the callback manager."""
        with self._lock:
            self._callback_manager.remove_handler(callback)
