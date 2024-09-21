"""Standard LangChain interface tests for ChatUnify"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_community.chat_models.unify import ChatUnify

@pytest.mark.requires("unify")
class TestChatUnifyStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUnify

    @property
    def chat_model_params(self) -> dict:
        return {"unify_api_key": "test_api_key"}

    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)

    @pytest.mark.asyncio
    async def test_async_generate(self, model: BaseChatModel) -> None:
        await super().test_async_generate(model)

    def test_streaming(self, model: BaseChatModel) -> None:
        super().test_streaming(model)

    @pytest.mark.asyncio
    async def test_async_streaming(self, model: BaseChatModel) -> None:
        await super().test_async_streaming(model)

    def test_custom_model(self) -> None:
        model = self.chat_model_class(
            unify_api_key="test_api_key",
            model="custom-model-name"
        )
        assert model.model == "custom-model-name"