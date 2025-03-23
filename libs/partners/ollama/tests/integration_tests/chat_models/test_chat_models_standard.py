"""Test chat model integration using standard integration tests."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_ollama.chat_models import ChatOllama


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3.1"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False
