from __future__ import annotations

from typing import List, Callable

from langchain.automaton.typedefs import MessageLog, MessageLike
from langchain.schema import PromptValue, BaseMessage


class MessageLogPromptValue(PromptValue):
    """Base abstract class for inputs to any language model."""

    message_log: MessageLog
    message_adapter: Callable[[MessageLike], List[BaseMessage]]

    class Config:
        arbitrary_types_allowed = True

    def to_string(self) -> str:
        """Return prompt value as string."""
        finalized = []
        for message in self.to_messages():
            prefix = message.type
            finalized.append(f"{prefix}: {message.content}")
        return "\n".join(finalized) + "\n" + "ai:"

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
        messages = []
        for message in self.message_log.messages:
            messages.extend(self.message_adapter(message))
        return messages

    @classmethod
    def from_message_log(
        cls,
        message_log: MessageLog,
        adapter: Callable[[MessageLike], List[BaseMessage]],
    ) -> MessageLogPromptValue:
        """Create a PromptValue from a MessageLog."""
        return cls(message_log=message_log, adapter=adapter)
