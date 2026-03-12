"""Standard LangChain interface tests."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,  # type: ignore[import-not-found]
)

from langchain_minimax import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


class TestMiniMaxStandard(ChatModelUnitTests):
    """Standard unit tests for `ChatMiniMax` chat model."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        """Chat model class being tested."""
        return ChatMiniMax

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {"model": MODEL_NAME}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "MINIMAX_API_KEY": "api_key",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "minimax_api_key": "api_key",
                "minimax_api_base": "https://api.minimax.io/v1",
            },
        )
