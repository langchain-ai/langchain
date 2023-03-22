from typing import List

from pydantic import BaseModel, Field

from langchain.schema import AIMessage, BaseMessage, HumanMessage, MessageStore


class DefaultMessageStore(MessageStore):
    """Using a list in the memory for message store"""

    # messages: List[BaseMessage] = Field(default_factory=list)
    messages: List[BaseMessage] = []
    session_id: str = "default"

    def read(self) -> List[BaseMessage]:
        return self.messages

    def add_user_message(self, message: HumanMessage) -> None:
        self.messages.append(message)

    def add_ai_message(self, message: AIMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages = []
