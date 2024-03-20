"""Test chat model integration."""


from langchain_cohere.chat_models import ChatCohere


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCohere(cohere_api_key="test")
