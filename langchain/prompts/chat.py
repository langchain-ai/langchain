from typing import Any
from abc import ABC, abstractmethod

from langchain.prompts.base import BasePromptTemplate
from langchain.schema import ChatMessage

class ChatPromptTemplate(BasePromptTemplate, ABC):


    def format(self, **kwargs: Any) -> str:
        return str(self.format_chat(**kwargs))

    @abstractmethod
    def format_chat(self, **kwargs) -> ChatMessage:
        """To be implemented by subclasses."""


    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError