"""Test chat model integration."""


from nvidia_aiplay.chat_models import ChatNVAIPlay


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatNVAIPlay()
