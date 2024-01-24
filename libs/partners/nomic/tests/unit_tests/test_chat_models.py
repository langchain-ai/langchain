"""Test chat model integration."""


from langchain_nomic.chat_models import ChatNomic


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatNomic()
