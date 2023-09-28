from abc import ABC, abstractmethod
from typing import Iterator, List

from langchain.schema.chat import ChatSession


class BaseChatLoader(ABC):
    """Base class for chat loaders."""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions."""

    def load(self) -> List[ChatSession]:
        """Eagerly load the chat sessions into memory."""
        return list(self.lazy_load())
