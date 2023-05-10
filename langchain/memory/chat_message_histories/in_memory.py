from typing import Any, List

from pydantic import BaseModel

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = []

    def add_user_message(self, message: str, **kwargs: Any) -> None:
        self.messages.append(HumanMessage(content=message, additional_kwargs=kwargs))

    def add_ai_message(self, message: str, **kwargs: Any) -> None:
        self.messages.append(AIMessage(content=message, additional_kwargs=kwargs))

    def clear(self) -> None:
        self.messages = []
