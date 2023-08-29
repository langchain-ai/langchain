from __future__ import annotations

import abc

import enum
from typing import Callable

from langchain.schema import (
    BaseMessage,
    FunctionMessage,
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseChatMessageHistory,
    PromptValue,
)


class MessageType(enum.Enum):
    """The type of message."""

    SYSTEM = enum.auto()
    USER = enum.auto()
    FUNCTION = enum.auto()
    AI = enum.auto()
    AI_INVOKE = enum.auto()
    AI_SELF = enum.auto()


def infer_message_type(message: BaseMessage) -> MessageType:
    """Infer the message type."""
    if isinstance(message, FunctionMessage):
        return MessageType.FUNCTION
    elif isinstance(message, AIMessage):
        if message.additional_kwargs:
            return MessageType.AI_INVOKE
        else:
            return MessageType.AI
    elif isinstance(message, SystemMessage):
        return MessageType.SYSTEM
    elif isinstance(message, HumanMessage):
        return MessageType.USER
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


class Memory(BaseChatMessageHistory):
    """A memory for the automaton."""

    def __init__(self, messages):
        self.messages = messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the memory."""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear the memory."""
        self.messages = []


# Interface that takes memory and returns a prompt value
PromptGenerator = Callable[[Memory], PromptValue]
