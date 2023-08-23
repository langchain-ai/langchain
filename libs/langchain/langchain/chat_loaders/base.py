from abc import ABC, abstractmethod
from typing import Iterator, List, Sequence, TypedDict

from langchain.schema.messages import BaseMessage


class ChatSession(TypedDict):
    """A chat session is a sequence of messages."""

    messages: Sequence[BaseMessage]


class BaseChatLoader(ABC):
    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions."""

    def load(self) -> List[ChatSession]:
        return list(self.lazy_load())
