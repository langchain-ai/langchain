"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_together import ChatTogether


class TestTogethertandard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatTogether

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
        }
