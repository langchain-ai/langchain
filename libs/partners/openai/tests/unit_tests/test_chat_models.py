"""Test chat model integration."""


from langchain_openai.chat_models import ChatOpenAI


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatOpenAI()
