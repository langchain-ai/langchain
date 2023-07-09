from __future__ import annotations

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ZepChatMessageHistory


class ZepMemory(ConversationBufferMemory):
    """Memory support for the Zep Long-term Memory Server"""

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)
