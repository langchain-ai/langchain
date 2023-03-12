from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_buffer_string
from langchain.schema import BaseLanguageModel, BaseMessage


class ConversationTokenBufferMemory(BaseChatMemory, BaseModel):
    """Buffer for storing conversation memory."""

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

    def get_num_tokens_list(self, arr: List[BaseMessage]) -> List[int]:
        """Get list of number of tokens in each string in the input array."""
        return [self.llm.get_num_tokens(get_buffer_string([x])) for x in arr]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        buffer: Any = self.buffer
        curr_buffer_length = sum(self.get_num_tokens_list(buffer))
        while curr_buffer_length > self.max_token_limit:
            buffer = buffer[1:]
            curr_buffer_length = sum(self.get_num_tokens_list(buffer))
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}
