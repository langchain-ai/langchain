"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_ai21 import ChatAI21


class TestAI21Standard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
            "api_key": "test_api_key",
        }
