"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_minimax.chat_models import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


class TestChatMiniMaxUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatMiniMax` chat model."""

    @property
    def chat_model_class(self) -> type[ChatMiniMax]:
        """Chat model class being tested."""
        return ChatMiniMax

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "MINIMAX_API_KEY": "api_key",
                "MINIMAX_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatMiniMax:
        """Get a chat model instance for testing."""
        return ChatMiniMax(**self.chat_model_params)


class TestChatMiniMaxCustomUnit:
    """Custom tests specific to MiniMax chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatMiniMax(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_default_model_name(self) -> None:
        """Test that model name is set correctly."""
        chat_model = ChatMiniMax(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.model_name == MODEL_NAME

    def test_llm_type(self) -> None:
        """Test that _llm_type returns the correct value."""
        chat_model = ChatMiniMax(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model._llm_type == "chat-minimax"


EXPECTED_MAX_INPUT_TOKENS = 204800


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatMiniMax(model="MiniMax-M2.5", api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["tool_calling"]
    assert model.profile["max_input_tokens"] == EXPECTED_MAX_INPUT_TOKENS
