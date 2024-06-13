"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_openai import AzureChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": "test",
            "openai_api_version": "2021-10-01",
            "azure_endpoint": "https://test.azure.com",
            "openai_api_key": "test",
        }
