from abc import ABC, abstractmethod
from typing import Iterator, List

from langchain_core.chat_sessions import ChatSession


class ChatLoaderInterface(ABC):
    """Interface for chat loaders."""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions."""

    @abstractmethod
    def load(self) -> List[ChatSession]:
        """Eagerly load the chat sessions into memory."""
