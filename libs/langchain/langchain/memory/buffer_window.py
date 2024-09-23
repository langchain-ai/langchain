from typing import Any, Dict, List, Union

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string

from langchain.memory.chat_memory import BaseChatMemory


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferWindowMemory(BaseChatMemory):
    """Use to keep track of the last k turns of a conversation.

    If the number of messages in the conversation is more than the maximum number
    of messages to keep, the oldest messages are dropped.
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    k: int = 5
    """Number of messages to store in buffer."""

    @property
    def buffer(self) -> Union[str, List[BaseMessage]]:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is False."""
        messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is True."""
        return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}
