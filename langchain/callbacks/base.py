"""Base callback handler that can be used to handle callbacks from langchain."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain.schema import AgentAction, LLMResult


class BaseCallbackHandler(ABC):
    """Base callback handler that can be used to handle callbacks from langchain."""

    @abstractmethod
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    @abstractmethod
    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """Run when LLM ends running."""

    @abstractmethod
    def on_llm_error(self, error: Exception) -> None:
        """Run when LLM errors."""

    @abstractmethod
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    @abstractmethod
    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Run when chain ends running."""

    @abstractmethod
    def on_chain_error(self, error: Exception) -> None:
        """Run when chain errors."""

    @abstractmethod
    def on_tool_start(
        self, serialized: Dict[str, Any], action: AgentAction, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    @abstractmethod
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    @abstractmethod
    def on_tool_error(self, error: Exception) -> None:
        """Run when tool errors."""

    @abstractmethod
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run when agent ends."""


class BaseCallbackManager(BaseCallbackHandler, ABC):
    """Base callback manager that can be used to handle callbacks from LangChain."""

    @abstractmethod
    def add_handler(self, callback: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""

    @abstractmethod
    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""

    @abstractmethod
    def set_handler(self, handler: BaseCallbackHandler) -> None:
        """Set handler as the only handler on the callback manager."""


class CallbackManager(BaseCallbackManager):
    """Callback manager that can be used to handle callbacks from langchain."""

    def __init__(self, handlers: List[BaseCallbackHandler]) -> None:
        """Initialize the callback manager."""
        self.handlers = handlers

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        for handler in self.handlers:
            handler.on_llm_start(serialized, prompts, **kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
    ) -> None:
        """Run when LLM ends running."""
        for handler in self.handlers:
            handler.on_llm_end(response)

    def on_llm_error(self, error: Exception) -> None:
        """Run when LLM errors."""
        for handler in self.handlers:
            handler.on_llm_error(error)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        for handler in self.handlers:
            handler.on_chain_start(serialized, inputs, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Run when chain ends running."""
        for handler in self.handlers:
            handler.on_chain_end(outputs)

    def on_chain_error(self, error: Exception) -> None:
        """Run when chain errors."""
        for handler in self.handlers:
            handler.on_chain_error(error)

    def on_tool_start(
        self, serialized: Dict[str, Any], action: AgentAction, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        for handler in self.handlers:
            handler.on_tool_start(serialized, action, **kwargs)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        for handler in self.handlers:
            handler.on_tool_end(output, **kwargs)

    def on_tool_error(self, error: Exception) -> None:
        """Run when tool errors."""
        for handler in self.handlers:
            handler.on_tool_error(error)

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on additional input from chains and agents."""
        for handler in self.handlers:
            handler.on_text(text, **kwargs)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)

    def set_handler(self, handler: BaseCallbackHandler) -> None:
        """Set handler as the only handler on the callback manager."""
        self.handlers = [handler]
