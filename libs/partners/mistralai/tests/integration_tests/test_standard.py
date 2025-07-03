"""Standard LangChain interface tests"""

from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_tests.integration_tests import \
    ChatModelIntegrationTests  # type: ignore[import-not-found]; type: ignore[import-not-found]


class TestMistralStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMistralAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "mistral-large-latest", "temperature": 0}

    @property
    def supports_json_mode(self) -> bool:
        return True
