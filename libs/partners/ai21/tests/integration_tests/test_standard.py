"""Standard LangChain interface tests"""

import time
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ai21 import ChatAI21


class TestAI21J2(ChatModelIntegrationTests):
    def teardown(self) -> None:
        # avoid getting rate limited
        time.sleep(1)

    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
        }


class TestAI21Jamba(ChatModelIntegrationTests):
    def teardown(self) -> None:
        # avoid getting rate limited
        time.sleep(1)

    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-instruct",
        }
