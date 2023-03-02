from abc import ABC
from typing import Any, List, Tuple

from langchain.prompts.base import BasePromptTemplate
from langchain.schema import ChatMessage


class ChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[Tuple[str, BasePromptTemplate]]

    def format(self, **kwargs: Any) -> str:
        return str(self.format_chat(**kwargs))

    def format_chat(self, **kwargs: Any) -> List[ChatMessage]:
        """To be implemented by subclasses."""
        result = []
        for role, prompt in self.messages:
            rel_params = {
                k: v for k, v in kwargs.items() if k in prompt.input_variables
            }
            message = prompt.format(**rel_params)
            result.append(ChatMessage(text=message, role=role))
        return result

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError
