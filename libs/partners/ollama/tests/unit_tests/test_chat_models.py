"""Test chat model integration."""

from typing import Dict, Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import ChatOllama


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> Dict:
        return {"model": "llama3-groq-tool-use"}
