"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_fireworks import ChatFireworks


class TestFireworksStandard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatFireworks

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "accounts/fireworks/models/firefunction-v1",
            "temperature": 0,
        }

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        super().test_tool_message_histories_list_content(
            chat_model_class, chat_model_params, chat_model_has_tool_calling
        )
