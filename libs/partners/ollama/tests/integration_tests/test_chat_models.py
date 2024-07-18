"""Test chat model integration."""

from langchain_ollama.chat_models import ChatOllama
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self):
        return ChatOllama

    @property
    def chat_model_params(self):
        return {"model": "llama3-groq-tool-use"}
