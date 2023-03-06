"""Chat prompt template."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Type, Union

from pydantic import BaseModel, Field

from langchain.prompts.base import BasePromptTemplate, StringPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    PromptValue,
    SystemMessage,
)


class BaseMessagePromptTemplate(BaseModel, ABC):
    prompt: StringPromptTemplate
    additional_kwargs: dict = Field(default_factory=dict)

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> BaseMessagePromptTemplate:
        prompt = PromptTemplate.from_template(template)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """To a BaseMessage."""


class ChatMessagePromptTemplate(BaseMessagePromptTemplate):
    role: str

    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


class HumanMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return HumanMessage(content=text, additional_kwargs=self.additional_kwargs)


class AIMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return AIMessage(content=text, additional_kwargs=self.additional_kwargs)


class SystemMessagePromptTemplate(BaseMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return SystemMessage(content=text, additional_kwargs=self.additional_kwargs)


class ChatPromptValue(PromptValue):
    messages: List[BaseMessage]

    def to_string(self) -> str:
        """Return prompt as string."""
        return str(self.messages)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.messages


class ChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[BaseMessagePromptTemplate]

    @classmethod
    def from_role_strings(
        cls, string_messages: List[Tuple[str, str]]
    ) -> ChatPromptTemplate:
        messages = [
            ChatMessagePromptTemplate(
                content=PromptTemplate.from_template(template), role=role
            )
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_strings(
        cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]
    ) -> ChatPromptTemplate:
        messages = [
            role(content=PromptTemplate.from_template(template))
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_messages(
        cls, messages: Sequence[BaseMessagePromptTemplate]
    ) -> ChatPromptTemplate:
        input_vars = set()
        for message in messages:
            input_vars.update(message.prompt.input_variables)
        return cls(input_variables=list(input_vars), messages=messages)

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        result = []
        for message_template in self.messages:
            rel_params = {
                k: v
                for k, v in kwargs.items()
                if k in message_template.prompt.input_variables
            }
            message = message_template.format(**rel_params)
            result.append(message)
        return ChatPromptValue(messages=result)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        raise NotImplementedError

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError

    def save(self, file_path: Union[Path, str]) -> None:
        raise NotImplementedError
