"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_ai21 import ChatAI21


class TestAI21J2(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
            "api_key": "test_api_key",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_standard_params(
            chat_model_class,
            chat_model_params,
        )


class TestAI21Jamba(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-instruct",
            "api_key": "test_api_key",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_standard_params(
            chat_model_class,
            chat_model_params,
        )
