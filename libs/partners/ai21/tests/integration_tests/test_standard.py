"""Standard LangChain interface tests"""

import time
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_ai21 import ChatAI21


class BaseTestAI21(ChatModelIntegrationTests):
    def teardown(self) -> None:
        # avoid getting rate limited
        time.sleep(1)

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @pytest.mark.xfail(reason="Not implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)


class TestAI21J2(BaseTestAI21):
    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
        }

    @pytest.mark.xfail(reason="Emits AIMessage instead of AIMessageChunk.")
    def test_stream(self, model: BaseChatModel) -> None:
        super().test_stream(model)

    @pytest.mark.xfail(reason="Emits AIMessage instead of AIMessageChunk.")
    async def test_astream(self, model: BaseChatModel) -> None:
        await super().test_astream(model)


class TestAI21Jamba(BaseTestAI21):
    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-instruct-preview",
        }
