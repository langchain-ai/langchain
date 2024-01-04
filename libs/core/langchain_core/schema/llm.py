from abc import ABC, abstractmethod
from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import LLMResult


# TODO: this is a work in progress
# TODO: do we need other methods from BaseLLM or BaseLanguageModel?
class LLMInterface(ABC):
    """Interface for LLM.

    The default implementation of this interface is :class:`BaseLLM`.
    The purpose of this class is to expose a simpler interface for
    LLMs, rather than expect the user to implement the full _generate method.
    """

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

    @abstractmethod
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

    @abstractmethod
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""

    @abstractmethod
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
