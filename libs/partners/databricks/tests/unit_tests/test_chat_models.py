"""Test chat model integration."""

from langchain_databricks.chat_models import ChatDatabricks


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatDatabricks()
