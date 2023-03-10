from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

import langchain
from langchain.base_language_model import BaseLanguageModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    LLMResult,
    PromptValue,
)


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseChatModel(BaseLanguageModel, BaseModel, ABC):
    """Base class for chat models."""

    def generate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Top Level call"""
        results = [self._generate(m, stop=stop) for m in messages]
        return LLMResult(generations=[res.generations for res in results])

    async def agenerate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Top Level call"""
        results = [await self._agenerate(m, stop=stop) for m in messages]
        return LLMResult(generations=[res.generations for res in results])

    def _generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop)

    async def _agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(prompt_messages, stop=stop)

    @abstractmethod
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    @abstractmethod
    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    def __call__(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> BaseMessage:
        return self._generate(messages, stop=stop).generations[0].message


class SimpleChatModel(BaseChatModel):
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        message = AIMessage(text=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> str:
        """Simpler interface."""
