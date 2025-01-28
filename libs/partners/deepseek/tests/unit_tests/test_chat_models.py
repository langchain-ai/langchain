"""Test chat model integration."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_deepseek.chat_models import ChatDeepSeek


class TestChatDeepSeekUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatDeepSeek]:
        return ChatDeepSeek

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "DEEPSEEK_API_KEY": "api_key",
            },
            {
                "model": "deepseek-chat",
            },
            {
                "api_key": "api_key",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "deepseek-chat",
        }
