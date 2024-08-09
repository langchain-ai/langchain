"""Standard LangChain interface tests"""
# requires either passing login information as environment variable,
# or passing into the chat_model_params

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_snowflake import ChatSnowflakeCortex

class TestChatSnowflakeCortex(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatSnowflakeCortex
    
    @property
    def chat_model_params(self) -> dict:
        return {
                "model": "mistral-large",
                "cortex_function": "complete",
        }

    @pytest.mark.xfail(
            reason=("Streaming not supported.")
    )
    def test_stream(self, model: BaseChatModel):
        super().test_stream(model)

    @pytest.mark.xfail(
            reason=("Streaming not supported.")
    )
    def test_astream(self, model: BaseChatModel):
        super().test_astream(model)

    @pytest.mark.xfail(
            reason=("Batch not supported.")
    )
    def test_batch(self, model: BaseChatModel):
        super().test_batch(model)

    @pytest.mark.xfail(
            reason=("Batch not supported.")
    )
    def test_abatch(self, model: BaseChatModel):
        super().test_abatch(model)

    @pytest.mark.xfail(
            reason=("Streaming not supported.")
    )
    def test_usage_metadata(self, model: BaseChatModel):
        super().test_usage_metadata(model)

    @pytest.mark.xfail(
            reason=("Streaming not supported.")
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel):
        super().test_usage_metadata_streaming(model)



