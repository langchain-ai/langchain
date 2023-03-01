from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain.schema import ChatGeneration, ChatResult


class BaseChat(ABC):
    def generate(
        self, messages: List[Dict], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""
        # Nothing here now, but future proofing.
        return self._generate(messages, stop=stop)

    @abstractmethod
    def _generate(
        self, messages: List[Dict], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""


class SimpleChat(BaseChat):
    role: str = "assistant"

    def _generate(
        self, messages: List[Dict], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        generation = ChatGeneration(text=output_str, role=self.role)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Simpler interface."""
