"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_hpc_ai.chat_models import DEFAULT_API_BASE, ChatHPCAI

MODEL_NAME = "minimax/minimax-m2.5"


class TestChatHPCAIUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatHPCAI` chat model."""

    @property
    def chat_model_class(self) -> type[ChatHPCAI]:
        """Chat model class being tested."""
        return ChatHPCAI

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "HPC_AI_API_KEY": "api_key",
                "HPC_AI_BASE_URL": "api_base",
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

    def get_chat_model(self) -> ChatHPCAI:
        """Get a chat model instance for testing."""
        return ChatHPCAI(**self.chat_model_params)


class TestChatHPCAICustomUnit:
    """Custom tests specific to HPC-AI chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatHPCAI(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_default_api_base(self) -> None:
        """Default base URL matches HPC-AI inference endpoint."""
        assert DEFAULT_API_BASE == "https://api.hpc-ai.com/inference/v1"
