"""Test chat model integration."""


from google.chat_models import ChatGoogleGenerativeAIChat


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatGoogleGenerativeAIChat()
