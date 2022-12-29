"""Base callback handler that can be used to handle callbacks from langchain."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain.schema import LLMResult


class CallbackHandler(ABC):
    """Base callback handler that can be used to handle callbacks from langchain."""

    @abstractmethod
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
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
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
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
        self, serialized: Dict[str, Any], action: str, tool_input: str, **extra: str
    ) -> None:
        """Run when tool starts running."""

    @abstractmethod
    def on_tool_end(self, output: str) -> None:
        """Run when tool ends running."""

    @abstractmethod
    def on_tool_error(self, error: Exception) -> None:
        """Run when tool errors."""
