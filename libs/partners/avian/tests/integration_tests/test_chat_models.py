"""Test ChatAvian chat model."""

from __future__ import annotations

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_avian.chat_models import ChatAvian

MODEL_NAME = "deepseek-v3.2"


class TestChatAvian(ChatModelIntegrationTests):
    """Test ``ChatAvian`` chat model."""

    @property
    def chat_model_class(self) -> type[ChatAvian]:
        """Return class of chat model being tested."""
        return ChatAvian

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }

    @property
    def supports_json_mode(self) -> bool:
        """Whether the chat model supports JSON mode."""
        return True
