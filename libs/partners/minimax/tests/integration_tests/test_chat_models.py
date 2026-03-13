"""Test ChatMiniMax chat model."""

from __future__ import annotations

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_minimax.chat_models import ChatMiniMax

MODEL_NAME = "MiniMax-M2.5"


class TestChatMiniMax(ChatModelIntegrationTests):
    """Test `ChatMiniMax` chat model."""

    @property
    def chat_model_class(self) -> type[ChatMiniMax]:
        """Return class of chat model being tested."""
        return ChatMiniMax

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0.5,
        }

    @property
    def supports_json_mode(self) -> bool:
        """Whether the chat model supports JSON mode."""
        return True
