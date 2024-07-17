"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_ai21 import ChatAI21


class TestAI21J2(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "j2-ultra",
            "api_key": "test_api_key",
        }


class TestAI21Jamba(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAI21

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "jamba-instruct",
            "api_key": "test_api_key",
        }
