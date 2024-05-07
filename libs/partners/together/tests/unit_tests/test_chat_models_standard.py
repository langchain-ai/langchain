"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_together import ChatTogether


class TestTogetherStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatTogether

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model": "meta-llama/Llama-3-8b-chat-hf",
        }
