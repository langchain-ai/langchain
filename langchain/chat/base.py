from langchain.chains.base import Chain
from abc import ABC
from langchain.memory.chat_memory import ChatMemory
from pydantic import root_validator
from typing import Dict

class BaseChatChain(Chain, ABC):

    human_prefix: str = "user"
    ai_prefix: str = "assistant"

    @root_validator(pre=True)
    def validate_memory_keys(cls, values: Dict) -> Dict:
        """Validate that the human and ai prefixes line up."""
        if "memory" in values:
            memory = values["memory"]
            if isinstance(memory, ChatMemory):
                if memory.human_prefix != values["human_prefix"]:
                    raise ValueError(
                        f"Memory human_prefix ({memory.human_prefix}) must "
                        f"match chain human_prefix ({values['human_prefix']})"
                    )
                if memory.ai_prefix != values["ai_prefix"]:
                    raise ValueError(
                        f"Memory ai_prefix ({memory.ai_prefix}) must "
                        f"match chain ai_prefix ({values['ai_prefix']})"
                    )
        return values

