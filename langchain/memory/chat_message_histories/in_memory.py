import json
from typing import List

from pydantic import BaseModel

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def clear(self) -> None:
        self.messages = []


class ActionBasedChatMessageHistory(ChatMessageHistory):
    """
    Chat message history that stores AI messages in the format the AI is expected to
    respond with. This reduces the frequency of the AI responding outside the response
    format (i.e. not in JSON).
    """

    def add_ai_message(self, message: str) -> None:
        message_obj = {"action": "Final Answer", "action_input": message}
        self.messages.append(AIMessage(content=json.dumps(message_obj)))
