"""Test chat model integration."""


from langchain_cloudflare.chat_models import ChatCloudflare


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCloudflare()
