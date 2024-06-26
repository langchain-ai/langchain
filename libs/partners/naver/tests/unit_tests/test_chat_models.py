"""Test chat model integration."""


from langchain_naver.chat_models import ChatNaver


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatNaver()
