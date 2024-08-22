import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field

from langchain.memory.utils import get_prompt_input_key


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(
        default_factory=InMemoryChatMessageHistory
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
            if len(outputs) == 1:
                output_key = list(outputs.keys())[0]
            elif "output" in outputs:
                output_key = "output"
                warnings.warn(
                    f"'{self.__class__.__name__}' got multiple output keys:"
                    f" {outputs.keys()}. The default 'output' key is being used."
                    f" If this is not desired, please manually set 'output_key'."
                )
            else:
                raise ValueError(
                    f"Got multiple output keys: {outputs.keys()}, cannot "
                    f"determine which to store in memory. Please set the "
                    f"'output_key' explicitly."
                )
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
