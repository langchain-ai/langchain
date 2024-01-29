"""Test chat model integration."""


from langchain_anthropic.chat_models import ChatAnthropicMessages


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatAnthropicMessages(model_name="claude-instant-1.2", anthropic_api_key="xyz")
    ChatAnthropicMessages(model="claude-instant-1.2", anthropic_api_key="xyz")
