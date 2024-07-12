"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_upstage import ChatUpstage


class TestUpstageStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUpstage

    @property
    def chat_model_params(self) -> dict:
        return {"model": "solar-1-mini-chat"}
