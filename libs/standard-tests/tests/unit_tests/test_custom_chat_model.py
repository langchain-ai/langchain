"""
Test the standard tests on the custom chat model in the docs
"""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_tests.unit_tests import ChatModelUnitTests

from .custom_chat_model import ChatParrotLink


class TestChatParrotLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_params(self) -> dict:
        return {"model": "bird-brain-001", "temperature": 0, "parrot_buffer_length": 50}


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatParrotLink]:
        return ChatParrotLink

    @property
    def chat_model_params(self) -> dict:
        return {"model": "bird-brain-001", "temperature": 0, "parrot_buffer_length": 50}
