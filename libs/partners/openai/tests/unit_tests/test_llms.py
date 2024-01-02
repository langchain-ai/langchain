"""Test OpenAI Chat API wrapper."""
from langchain_openai import OpenAI


def test_initialization() -> None:
    """Test integration initialization."""
    OpenAI()
