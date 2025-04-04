"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
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

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "FIREWORKS_API_KEY": "api_key",
                "FIREWORKS_API_BASE": "https://base.com",
            },
            {},
            {
                "fireworks_api_key": "api_key",
                "fireworks_api_base": "https://base.com",
            },
        )
