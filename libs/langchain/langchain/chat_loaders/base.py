from abc import ABC, abstractmethod
from typing import Iterator, List, TypedDict

from langchain.schema.messages import BaseMessage


class ChatSession(TypedDict):
    """A chat session is a sequence of messages."""

    messages: List[BaseMessage]


class BaseChatLoader(ABC):
    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the messages from the chat file and yield them in the required format."""

    def load(self) -> List[ChatSession]:
        return list(self.lazy_load())
