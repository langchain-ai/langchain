"""Test chat model integration."""


from langchain_together.chat_models import ChatTogether


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatTogether()
