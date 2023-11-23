"""Memory used to save agent output AND intermediate steps."""
from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string

from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.memory.chat_memory import BaseChatMemory
from langchain.pydantic_v1 import PrivateAttr
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)


class AgentTokenBufferMemory(BaseChatMemory):
    """Memory used to save agent output AND intermediate steps."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 12000
    """The max number of tokens to keep in the buffer.
    Once the buffer exceeds this many tokens, the oldest messages will be pruned."""
    return_messages: bool = True
    output_key: str = "output"
    intermediate_steps_key: str = "intermediate_steps"

    _chat_buffer: List[BaseMessage] = PrivateAttr(default_factory=list)
    """The local chat buffer that holds latest messages whose total token size
    does not exceed max_token_limit"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # fill in _chat_buffer with messages from chat_history until it reaches
        # max token limit
        temp_chat_buffer = []
        chat_buffer_length = 0
        for m in reversed(self.chat_memory.messages):
            m_length = self.llm.get_num_tokens(m.content)
            if chat_buffer_length + m_length <= self.max_token_limit:
                temp_chat_buffer.append(m)
                chat_buffer_length += m_length
            else:
                break
        self._chat_buffer = list(reversed(temp_chat_buffer))

    @property
    def buffer(self) -> List[BaseMessage]:
        """Message buffer for the chat."""
        return self._chat_buffer

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        if self.return_messages:
            final_buffer: Any = self.buffer
        else:
            final_buffer = get_buffer_string(
                self.buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context from this conversation to buffer, making sure the total
        buffer size does not go over specified token limit"""

        input_str, output_str = self._get_input_output(inputs, outputs)
        self._add_user_message(input_str)
        steps = format_to_openai_function_messages(outputs[self.intermediate_steps_key])
        for msg in steps:
            self._add_message(msg)
        self._add_ai_message(output_str)

    def clear(self) -> None:
        """Clear buffer contents but keep history intact."""
        self._chat_buffer.clear()

    def _add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to memory and history.

        Args:
            message: The string contents of a human message.
        """
        m = HumanMessage(content=message)
        self._add_message(m)

    def _add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to memory and history.

        Args:
            message: The string contents of an AI message.
        """
        m = AIMessage(content=message)
        self._add_message(m)

    def _add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the memory and history.

        Args:
            message: A BaseMessage object to memory and history.
        """
        # store message in the chat_memory for persistence
        self.chat_memory.add_message(message)

        # prune chat buffer until there is space for new message or the buffer is empty
        # then add the new message if there is space for it in the buffer
        m_length = self.llm.get_num_tokens(message.content)
        chat_buffer_length = self.llm.get_num_tokens_from_messages(self._chat_buffer)
        while (self.max_token_limit - chat_buffer_length < m_length) and len(
            self._chat_buffer
        ) > 0:
            self._chat_buffer.pop(0)
            chat_buffer_length = self.llm.get_num_tokens_from_messages(
                self._chat_buffer
            )

        if self.max_token_limit - chat_buffer_length >= m_length:
            self._chat_buffer.append(message)
