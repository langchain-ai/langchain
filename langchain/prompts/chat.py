"""Chat prompt template."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Type, Union

from pydantic import BaseModel, Field

from langchain.prompts.base import BasePromptTemplate, PromptValue
from langchain.prompts.prompt import BaseStringPromptTemplate, StringPromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


class BaseMessagePromptTemplate(BaseModel, ABC):
    prompt: BaseStringPromptTemplate
    additional_kwargs: dict = Field(default_factory=dict)

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> BaseMessagePromptTemplate:
        prompt = StringPromptTemplate.from_template(template)
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


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """Base class for chat prompt templates."""

    @abstractmethod
    def format(self, **kwargs: Any) -> Sequence[BaseMessage]:
        """Format to a sequence of BaseMessages."""

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format to a ChatPromptValue."""
        return ChatPromptValue(messages=self.format(**kwargs))


class ChatPromptTemplate(BaseChatPromptTemplate, ABC):
    """Chat prompt template."""

    messages: List[BaseMessagePromptTemplate]

    @classmethod
    def from_role_strings(
        cls, string_messages: List[Tuple[str, str]]
    ) -> BaseChatPromptTemplate:
        messages = [
            ChatMessagePromptTemplate(
                content=StringPromptTemplate.from_template(template), role=role
            )
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_strings(
        cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]
    ) -> BaseChatPromptTemplate:
        messages = [
            role(content=StringPromptTemplate.from_template(template))
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_messages(
        cls, messages: Sequence[BaseMessagePromptTemplate]
    ) -> BaseChatPromptTemplate:
        input_vars = set()
        for message in messages:
            input_vars.update(message.prompt.input_variables)
        return cls(input_variables=list(input_vars), messages=messages)

    def format(self, **kwargs: Any) -> Sequence[BaseMessage]:
        """Format to a sequence of BaseMessages."""
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
