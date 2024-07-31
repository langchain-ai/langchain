"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_community.chat_models.litellm import ChatLiteLLM


class TestLiteLLMStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        return {"model": "ollama/mistral"}

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)
