"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_community.chat_models.litellm import ChatLiteLLM


@pytest.mark.requires("litellm")
class TestLiteLLMStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        return {"api_key": "test_api_key"}

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)
