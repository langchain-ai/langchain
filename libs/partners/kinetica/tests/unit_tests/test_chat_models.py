"""Test chat model integration."""


from langchain_kinetica.chat_models import ChatKinetica


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatKinetica()
