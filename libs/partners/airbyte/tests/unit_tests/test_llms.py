"""Test Airbyte Chat API wrapper."""
from langchain_airbyte import AirbyteLLM


def test_initialization() -> None:
    """Test integration initialization."""
    AirbyteLLM()
