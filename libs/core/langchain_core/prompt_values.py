from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Sequence

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""


class StringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    """Prompt text."""
    type: Literal["StringPromptValue"] = "StringPromptValue"

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=self.text)]


class ChatPromptValue(PromptValue):
    """Chat prompt value.

    A type of a prompt value that is built from messages.
    """

    messages: Sequence[BaseMessage]
    """List of messages."""

    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of messages."""
        return list(self.messages)


class ChatPromptValueConcrete(ChatPromptValue):
    """Chat prompt value which explicitly lists out the message types it accepts.
    For use in external schemas."""

    messages: Sequence[AnyMessage]

    type: Literal["ChatPromptValueConcrete"] = "ChatPromptValueConcrete"
