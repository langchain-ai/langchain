from typing import List

from pydantic import BaseModel

from langchain.schema import (
    BaseChatMessageHistory,
    BaseMessage,
)


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []
