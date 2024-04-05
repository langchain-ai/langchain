"""Test chat model integration."""


from langchain_upstage.chat_models import ChatUpstage


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatUpstage()
