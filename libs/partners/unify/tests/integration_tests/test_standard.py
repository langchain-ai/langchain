"""Standard LangChain interface tests"""
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_unify import ChatUnify


class TestUnify(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUnify

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "test-model",
            "unify_api_key": "test-api-key",
        }

    @pytest.fixture
    def test_stream(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_stream(
            chat_model_class,
            chat_model_params,
        )

    @pytest.fixture
    async def test_astream(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        await super().test_astream(
            chat_model_class,
            chat_model_params,
        )