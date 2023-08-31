from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, List, Mapping

from langchain.schema import (
    BaseMessage,
    PromptValue,
)


@dataclasses.dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: Optional[Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class FunctionResult:
    result: Any
    error: Optional[str]


@dataclasses.dataclass(frozen=True)
class AgentFinish:
    result: Any


MessageLike = BaseMessage | FunctionCall | FunctionResult | AgentFinish


class MessageLog:
    """A generalized message log for message like items."""

    def __init__(self, messages: Sequence[MessageLike]):
        self.messages = list(messages)

    def add_messages(self, messages: Sequence[MessageLike]):
        self.messages.extend(messages)


class MessageLogPromptValue(PromptValue):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    message_log: MessageLog

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
        return [
            message
            for message in self.message_log.messages
            if isinstance(message, BaseMessage)
        ]
