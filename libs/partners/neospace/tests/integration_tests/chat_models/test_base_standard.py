"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_neospace import ChatNeoSpace


class TestNeoSpaceStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatNeoSpace

    @property
    def chat_model_params(self) -> dict:
        return {"model": "neo-4o", "stream_usage": True}

    @property
    def supports_image_inputs(self) -> bool:
        return True
