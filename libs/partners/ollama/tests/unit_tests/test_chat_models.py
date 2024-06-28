"""Test chat model integration."""


from langchain_ollama.chat_models import ChatOllama


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatOllama()
