from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field, validator

import langchain
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import BaseMessage, ChatGeneration, ChatResult, LLMResult


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseChatModel(BaseModel, ABC):
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

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
