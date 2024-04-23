"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_fireworks import ChatFireworks


class TestFireworksStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatFireworks

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "api_key": "test_api_key",
        }
