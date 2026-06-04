"""Standard LangChain interface tests."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests.chat_models import ChatModelUnitTests

from langchain_ollama import ChatOllama


class TestOllamaStandard(ChatModelUnitTests):
    """Run ChatOllama on LangChain standard tests."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3.2"}
