"""Test chat model integration."""


from google.chat_models import ChatGoogleGenerativeAI


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatGoogleGenerativeAI()
