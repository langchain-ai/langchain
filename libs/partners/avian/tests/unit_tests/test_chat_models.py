"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_avian.chat_models import ChatAvian

MODEL_NAME = "deepseek-v3.2"


class TestChatAvianUnit(ChatModelUnitTests):
    """Standard unit tests for ``ChatAvian`` chat model."""

    @property
    def chat_model_class(self) -> type[ChatAvian]:
        """Chat model class being tested."""
        return ChatAvian

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "AVIAN_API_KEY": "api_key",
                "AVIAN_API_BASE": "api_base",
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

    def get_chat_model(self) -> ChatAvian:
        """Get a chat model instance for testing."""
        return ChatAvian(**self.chat_model_params)


class TestChatAvianCustomUnit:
    """Custom tests specific to Avian chat model."""

    def test_base_url_alias(self) -> None:
        """Test that ``base_url`` is accepted as an alias for ``api_base``."""
        chat_model = ChatAvian(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_default_api_base(self) -> None:
        """Test that the default API base is set correctly."""
        chat_model = ChatAvian(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.api_base == "https://api.avian.io/v1"

    def test_llm_type(self) -> None:
        """Test that the LLM type is correct."""
        chat_model = ChatAvian(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model._llm_type == "chat-avian"

    def test_lc_secrets(self) -> None:
        """Test that secrets map is correct."""
        chat_model = ChatAvian(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.lc_secrets == {"api_key": "AVIAN_API_KEY"}

    def test_model_profile(self) -> None:
        """Test that model profile is loaded correctly."""
        model = ChatAvian(model="deepseek-v3.2", api_key=SecretStr("test_key"))
        assert model.profile is not None
        assert model.profile["max_input_tokens"] == 164000

    def test_model_profile_minimax(self) -> None:
        """Test that MiniMax M2.5 profile has 1M context."""
        model = ChatAvian(model="minimax-m2.5", api_key=SecretStr("test_key"))
        assert model.profile is not None
        assert model.profile["max_input_tokens"] == 1000000
