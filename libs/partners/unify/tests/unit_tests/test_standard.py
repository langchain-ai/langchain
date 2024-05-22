"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_unify import ChatUnify


class TestUnify(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUnify

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "test-model",
            "unify_api_key": "test_api_key",
        }

    @pytest.fixture
    def test_standard_params(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_standard_params(
            chat_model_class,
            chat_model_params,
        )