"""Test chat model integration."""


from langchain_robocorp.chat_models import ChatActionServer


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatActionServer()
