"""Test GoogleGenerativeAIChat Chat API wrapper."""
from google import GoogleGenerativeAIChatLLM


def test_integration_initialization() -> None:
    """Test integration initialization."""
    GoogleGenerativeAIChatLLM()
