from typing import List

from pydantic import BaseModel

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []

    def partial_clear(self, delete_ratio: float = 0.5) -> None:
        """Clear a portion of the history."""
        self.messages = self.messages[-1 * int(len(self.messages) * delete_ratio) :]
