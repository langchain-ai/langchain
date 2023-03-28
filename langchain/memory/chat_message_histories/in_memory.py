from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)


class ChatMessageHistory(BaseChatMessageHistory):
    messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []
