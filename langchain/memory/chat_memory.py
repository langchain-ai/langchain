from pydantic import BaseModel, Field
from typing import List
from langchain.schema import ChatGeneration

class ChatMemory(BaseModel):
    human_prefix: str = "user"
    ai_prefix: str = "assistant"
    messages: List[ChatGeneration] = Field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        gen = ChatGeneration(text=message, role=self.human_prefix)
        self.messages.append(gen)

    def add_ai_message(self, message: str) -> None:
        gen = ChatGeneration(text=message, role=self.ai_prefix)
        self.messages.append(gen)

    def clear(self):
        self.messages = []
