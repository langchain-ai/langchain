"""Standard LangChain interface tests"""

from langchain_core.language_models import BaseChatModel
from langchain_fireworks import ChatFireworks
from langchain_tests.unit_tests import \
    ChatModelUnitTests  # type: ignore[import-not-found]; type: ignore[import-not-found]


class TestFireworksStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatFireworks

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "api_key": "test_api_key",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "FIREWORKS_API_KEY": "api_key",
                "FIREWORKS_API_BASE": "https://base.com",
            },
            {
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            },
            {
                "fireworks_api_key": "api_key",
                "fireworks_api_base": "https://base.com",
            },
        )
