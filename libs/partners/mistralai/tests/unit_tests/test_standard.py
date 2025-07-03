"""Standard LangChain interface tests"""

from langchain_core.language_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_tests.unit_tests import \
    ChatModelUnitTests  # type: ignore[import-not-found]; type: ignore[import-not-found]


class TestMistralStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMistralAI
