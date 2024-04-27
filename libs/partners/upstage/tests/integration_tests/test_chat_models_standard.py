"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_upstage import ChatUpstage


class TestUpstageStandard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUpstage

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "solar-1-mini-chat",
        }
