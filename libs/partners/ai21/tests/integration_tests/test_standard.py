"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ai21 import ChatAI21


class TestAI21Standard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
        }

    @pytest.mark.xfail(reason="Emits AIMessage instead of AIMessageChunk.")
    def test_stream(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_stream(
            chat_model_class,
            chat_model_params,
        )

    @pytest.mark.xfail(reason="Emits AIMessage instead of AIMessageChunk.")
    async def test_astream(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        await super().test_astream(
            chat_model_class,
            chat_model_params,
        )
