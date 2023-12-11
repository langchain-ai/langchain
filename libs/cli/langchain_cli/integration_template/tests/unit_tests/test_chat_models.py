"""Test chat model integration."""


from langchain_integration.chat_models import ChatIntegration


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatIntegration()
