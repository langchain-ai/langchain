from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.schema import ChatGeneration, ChatMessage, ChatResult


class BaseChatModel(ABC):
    def generate(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""
        # Nothing here now, but future proofing.
        return self._generate(messages, stop=stop)

    async def agenerate(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        raise NotImplementedError

    @abstractmethod
    def _generate(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    def run(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> ChatMessage:
        res = self.generate(messages, stop=stop)
        return res.generations[0].message


class SimpleChatModel(BaseChatModel):
    role: str = "assistant"

    def _generate(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        message = ChatMessage(text=output_str, role=self.role)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self, messages: List[ChatMessage], stop: Optional[List[str]] = None
    ) -> str:
        """Simpler interface."""
