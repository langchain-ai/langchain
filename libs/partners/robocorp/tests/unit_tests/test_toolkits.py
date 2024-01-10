"""Test toolkit integration."""
from langchain_robocorp.toolkits import ActionServerToolkit

from ._fixtures import FakeChatLLMT


def test_initialization() -> None:
    """Test toolkit initialization."""
    ActionServerToolkit(url="http://localhost", llm=FakeChatLLMT())
