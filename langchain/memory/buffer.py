from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.memory.chat_memory import ChatMemoryMixin
from langchain.memory.utils import get_buffer_string


class ConversationBufferMemory(ChatMemoryMixin, BaseModel):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> str:
        """String buffer of memory."""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}
