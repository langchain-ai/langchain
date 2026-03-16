"""Standard LangChain integration tests for ChatSarvam."""

from typing import Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_sarvamcloud.chat_models import ChatSarvam


@pytest.mark.requires("SARVAM_API_SUBSCRIPTION_KEY")
class TestChatSarvamStandardIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSarvam]:
        return ChatSarvam

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "sarvam-105b",
            "temperature": 0.0,
            "max_tokens": 100,
        }
