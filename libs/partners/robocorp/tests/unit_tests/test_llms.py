"""Test ActionServer Chat API wrapper."""
from langchain_robocorp import ActionServerLLM


def test_initialization() -> None:
    """Test integration initialization."""
    ActionServerLLM()
