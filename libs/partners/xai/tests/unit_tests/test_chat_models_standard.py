"""Standard LangChain interface tests"""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_xai import ChatXAI


class TestXAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatXAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "grok-beta"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "XAI_API_KEY": "api_key",
            },
            {
                "model": "grok-beta",
            },
            {
                "xai_api_key": "api_key",
                "xai_api_base": "https://api.x.ai/v1/",
            },
        )
