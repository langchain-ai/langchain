"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_fireworks import ChatFireworks


class TestFireworksStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatFireworks

    @property
    def chat_model_params(self) -> dict:
        return {"api_key": "test_api_key"}
