"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_together import ChatTogether


class TestTogetherStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatTogether

    @property
    def chat_model_params(self) -> dict:
        return {"model": "mistralai/Mistral-7B-Instruct-v0.1"}
