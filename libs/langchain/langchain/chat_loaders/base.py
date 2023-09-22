from abc import ABC, abstractmethod
from typing import Iterator, List, Sequence, TypedDict

from langchain.schema.messages import BaseMessage


class ChatSession(TypedDict):
    """Chat Session represents a single
    conversation, channel, or other group of messages."""

    messages: Sequence[BaseMessage]
    """The LangChain chat messages loaded from the source."""


class BaseChatLoader(ABC):
    """Base class for chat loaders."""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions."""

    def load(self) -> List[ChatSession]:
        """Eagerly load the chat sessions into memory."""
        return list(self.lazy_load())
