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
