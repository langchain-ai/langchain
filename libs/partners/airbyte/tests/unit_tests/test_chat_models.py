"""Test chat model integration."""


from langchain_airbyte.chat_models import ChatAirbyte


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatAirbyte()
