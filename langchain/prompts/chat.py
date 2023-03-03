from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Tuple, Type, Union

from pydantic import BaseModel

from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


class BaseMessagePromptTemplate(BaseModel, ABC):
    prompt: BasePromptTemplate

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """To a BaseMessage."""


class ChatMessagePromptTemplate(BaseMessagePromptTemplate):
    role: str

    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return ChatMessage(text=text, role=self.role)


class HumanMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return HumanMessage(text=text)


class AIMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return AIMessage(text=text)


class SystemMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return SystemMessage(text=text)


class ChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[BaseMessagePromptTemplate]

    @classmethod
    def from_role_strings(
        cls, string_messages: List[Tuple[str, str]]
    ) -> ChatPromptTemplate:
        messages = [
            ChatMessagePromptTemplate(
                text=PromptTemplate.from_template(template), role=role
            )
            for role, template in string_messages
        ]
        input_vars = set([m.prompt.input_variables] for m in messages)
        return cls(input_variables=list(input_vars), messages=messages)

    @classmethod
    def from_strings(
        cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]
    ) -> ChatPromptTemplate:
        messages = [
            role(text=PromptTemplate.from_template(template))
            for role, template in string_messages
        ]
        input_vars = set([m.prompt.input_variables] for m in messages)
        return cls(input_variables=list(input_vars), messages=messages)

    def format(self, **kwargs: Any) -> str:
        return str(self.format_chat(**kwargs))

    def format_chat(self, **kwargs: Any) -> List[BaseMessage]:
        """Format message templates."""
        result = []
        for message_template in self.messages:
            rel_params = {
                k: v
                for k, v in kwargs.items()
                if k in message_template.prompt.input_variables
            }
            message = message_template.format(**rel_params)
            result.append(message)
        return result

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        raise NotImplementedError

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError

    def save(self, file_path: Union[Path, str]) -> None:
        raise NotImplementedError
