"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_anthropic import ChatAnthropic


class TestAnthropicStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAnthropic

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "claude-3-haiku-20240307",
        }
