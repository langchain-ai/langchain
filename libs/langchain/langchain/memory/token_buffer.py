from typing import Any, Dict, List

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, get_buffer_string


class ConversationTokenBufferMemory(BaseChatMemory):
    """Conversation chat memory with token limit."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages.copy()
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        while curr_buffer_length > self.max_token_limit:
            buffer.pop(0)
            curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}
