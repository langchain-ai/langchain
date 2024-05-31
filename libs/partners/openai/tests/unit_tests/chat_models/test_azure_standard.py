"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_openai import AzureChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": "test",
            "openai_api_version": "2021-10-01",
            "azure_endpoint": "https://test.azure.com",
        }
