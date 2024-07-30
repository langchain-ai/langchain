"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_mistralai import ChatMistralAI


class TestMistralStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatMistralAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "mistral-large-latest", "temperature": 0}
