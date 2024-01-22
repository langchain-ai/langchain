"""Test chat model integration."""


from langchain_pinecone.chat_models import ChatPinecone


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatPinecone()
