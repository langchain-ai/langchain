from typing import List

from pydantic import BaseModel, Field

from langchain_core.chat_history import (
    BaseChatMessageHistory,
)
from langchain_core.messages import BaseMessage


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        if not isinstance(message, BaseMessage):
            raise ValueError
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []
