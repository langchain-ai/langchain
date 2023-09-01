from __future__ import annotations

import json
from typing import List

from langchain.automaton.typedefs import MessageLog, FunctionResult
from langchain.schema import PromptValue, BaseMessage, FunctionMessage


class MessageLogPromptValue(PromptValue):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    message_log: MessageLog
    # If True will use the OpenAI function method
    use_function_message: bool = (
        False  # TODO(Eugene): replace with adapter, should be generic
    )

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
            if isinstance(message, BaseMessage):
                messages.append(message)
            elif isinstance(message, FunctionResult):
                if self.use_function_message:
                    messages.append(
                        FunctionMessage(
                            name=message.name, content=json.dumps(message.result)
                        )
                    )
            else:
                # Ignore internal messages
                pass
        return messages

    @classmethod
    def from_message_log(cls, message_log: MessageLog) -> MessageLogPromptValue:
        """Create a PromptValue from a MessageLog."""
        return cls(message_log=message_log)
