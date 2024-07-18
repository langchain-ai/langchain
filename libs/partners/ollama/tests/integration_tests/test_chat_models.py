"""Test chat model integration."""

from typing import Type

from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ollama.chat_models import ChatOllama


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3-groq-tool-use"}
