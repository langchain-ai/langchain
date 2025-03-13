"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests.chat_models import (
    ChatModelUnitTests,
)

from langchain_groq import ChatGroq


class TestGroqStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama-3.1-8b-instant"}
