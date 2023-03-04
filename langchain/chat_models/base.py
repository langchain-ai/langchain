from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.schema import BaseMessage, ChatGeneration, ChatResult, LLMResult


class BaseChatModel(ABC):
    def generate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Top Level call"""
        results = []
        for m in messages:
            results.append(self._generate(m, stop=stop))
        return LLMResult(generations=[res.generations for res in results])

    async def agenerate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        raise NotImplementedError

    @abstractmethod
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    def run(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> BaseMessage:
        return self._generate(messages, stop=stop).generations[0].message


class SimpleChatModel(BaseChatModel):
    role: str = "assistant"

    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        message = BaseMessage(text=output_str, role=self.role)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> str:
        """Simpler interface."""
