from typing import List

from pydantic import BaseModel, Field

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    chat_messages: List[BaseMessage] = Field(default_factory=list, alias="messages")

    @property
    def messages(self) -> List[BaseMessage]:
        return self.chat_messages

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.chat_messages = []
