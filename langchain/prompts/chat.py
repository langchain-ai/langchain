from __future__ import annotations

from abc import ABC
from typing import Any, Callable, List, Tuple, Union

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import ChatMessage


class ChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[Tuple[str, BasePromptTemplate]]

    @classmethod
    def from_strings(cls, string_messages: List[Tuple[str, str]]) -> ChatPromptTemplate:
        messages = [
            (role, PromptTemplate.from_template(template))
            for role, template in string_messages
        ]
        input_vars = set([m.input_variables] for _, m in messages)
        return cls(input_variables=list(input_vars), messages=messages)

    def format(self, **kwargs: Any) -> str:
        return str(self.format_chat(**kwargs))

    def format_chat(self, **kwargs: Any) -> List[ChatMessage]:
        """Format message templates."""
        result = []
        for role, prompt in self.messages:
            rel_params = {
                k: v for k, v in kwargs.items() if k in prompt.input_variables
            }
            message = prompt.format(**rel_params)
            result.append(ChatMessage(text=message, role=role))
        return result

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        raise NotImplementedError

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError
