from typing import List

from pydantic import BaseModel, Field

from langchain.schema import AIMessage, BaseMessage, HumanMessage


class ChatMessageHistory(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []
