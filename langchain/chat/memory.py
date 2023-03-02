from typing import Dict, Any, List

from langchain.chains.base import Memory
from langchain.memory.chat_memory import ChatMemory
from abc import ABC, abstractmethod

def _get_prompt_input_key(inputs: Dict[str, Any]) -> str:
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]

class SimpleChatMemory(Memory, ChatMemory):
    def clear(self) -> None:
        self.clear()

    @property
    def memory_variables(self) -> List[str]:
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        self.add_user_message(inputs[_get_prompt_input_key(inputs)])
        self.add_ai_message(outputs[_get_prompt_input_key(outputs)])
