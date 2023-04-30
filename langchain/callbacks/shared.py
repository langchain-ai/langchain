"""A shared CallbackManager."""

import threading
from typing import Any, Dict, List, Union

from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    CallbackManager,
)
from langchain.schema import AgentAction, AgentFinish, LLMResult


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

    _callback_manager: CallbackManager = CallbackManager(handlers=[])

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        with self._lock:
            self._callback_manager.on_llm_start(serialized, prompts, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        with self._lock:
            self._callback_manager.on_llm_end(response, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        with self._lock:
            self._callback_manager.on_llm_new_token(token, **kwargs)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        with self._lock:
            self._callback_manager.on_llm_error(error, **kwargs)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        with self._lock:
            self._callback_manager.on_chain_start(serialized, inputs, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        with self._lock:
            self._callback_manager.on_chain_end(outputs, **kwargs)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""
        with self._lock:
            self._callback_manager.on_chain_error(error, **kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        with self._lock:
            self._callback_manager.on_tool_start(serialized, input_str, **kwargs)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        with self._lock:
            self._callback_manager.on_agent_action(action, **kwargs)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        with self._lock:
            self._callback_manager.on_tool_end(output, **kwargs)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        with self._lock:
            self._callback_manager.on_tool_error(error, **kwargs)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        with self._lock:
            self._callback_manager.on_text(text, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        with self._lock:
            self._callback_manager.on_agent_finish(finish, **kwargs)

    def add_handler(self, callback: BaseCallbackHandler) -> None:
        """Add a callback to the callback manager."""
        with self._lock:
            self._callback_manager.add_handler(callback)

    def remove_handler(self, callback: BaseCallbackHandler) -> None:
        """Remove a callback from the callback manager."""
        with self._lock:
            self._callback_manager.remove_handler(callback)

    def set_handlers(self, handlers: List[BaseCallbackHandler]) -> None:
        """Set handlers as the only handlers on the callback manager."""
        with self._lock:
            self._callback_manager.handlers = handlers
