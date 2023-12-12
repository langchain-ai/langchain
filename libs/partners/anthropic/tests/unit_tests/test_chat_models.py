"""Test chat model integration."""


from langchain_anthropic.chat_models import ChatAnthropic


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatAnthropic()
