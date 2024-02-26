"""Test chat model integration."""


from langchain_voyageai.chat_models import ChatVoyageAI


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatVoyageAI()
