from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_buffer_string


class ConversationBufferMemory(BaseChatMemory, BaseModel):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    buffer: str = ""

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        self.buffer += "\n" + get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        if self.return_messages:
            buffer: Any = self.chat_memory.messages
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

    def clear(self) -> None:
        super().clear()
        self.buffer = ""
