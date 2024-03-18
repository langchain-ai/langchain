from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field

from langchain.memory.utils import get_prompt_input_key


def _chat_memory_default_factory() -> BaseChatMessageHistory:
    replacement = """
!pip install -U langchain-community

from langchain_community.chat_message_histories import ChatMessageHistory

chat_history = ChatMessageHistory()
memory = BaseChatMemory(chat_memory=chat_history, ...)
"""
    raise ValueError(
        "Must specify chat_memory: BaseChatMessageHistory. Example:"
        f"```{replacement}```"
    )


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(
        default_factory=_chat_memory_default_factory
    )
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        await self.chat_memory.aadd_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    async def aclear(self) -> None:
        """Clear memory contents."""
        await self.chat_memory.aclear()
