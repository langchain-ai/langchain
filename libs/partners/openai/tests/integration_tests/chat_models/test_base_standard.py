"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import ChatOpenAI


class TestOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI
