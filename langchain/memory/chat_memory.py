from abc import ABC
from typing import Any, Dict, Optional, Tuple

from pydantic import Field

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory, BaseMemory


class BaseChatMemory(BaseMemory, ABC):
    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        prompt_input_key = self.get_prompt_input_key(inputs)
        output_key = self.get_prompt_output_key(outputs)
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    def get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """
        Get the prompt input key.

        Args:
            inputs: Dict[str, Any]

        Returns:
            A prompt input key.
        """
        if self.input_key is not None:
            return self.input_key

        # "stop" is a special key that can be passed as input but is not used to
        # format the prompt.
        prompt_input_variables = list(set(inputs).difference(self.memory_variables + ["stop"]))
        if len(prompt_input_variables) != 1:
            raise ValueError(f"Missing input_key arg with multiple prompt input variables: {prompt_input_variables}")
        return prompt_input_variables[0]

    def get_prompt_output_key(self, outputs: Dict[str, str]) -> str:
        """
        Get the prompt output key.

        Args:
            outputs: Dict[str, Any]

        Returns:
            A prompt output key.
        """
        if self.output_key is not None:
            return self.output_key

        if len(outputs) != 1:
            raise ValueError(f"Missing output_key arg with multiple prompt output variables: {outputs.keys()}")
        return list(outputs.keys())[0]