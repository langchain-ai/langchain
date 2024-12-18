"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_fireworks import ChatFireworks


class TestFireworksStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatFireworks

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "accounts/fireworks/models/firefunction-v2",
            "temperature": 0,
        }

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @property
    def supports_json_mode(self) -> bool:
        return True
