"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_ai21 import ChatAI21


class TestAI21J2(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
            "api_key": "test_api_key",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)


class TestAI21Jamba(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-instruct",
            "api_key": "test_api_key",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)
