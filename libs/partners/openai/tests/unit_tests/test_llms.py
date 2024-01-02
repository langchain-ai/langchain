"""Test OpenAI Chat API wrapper."""
from langchain_openai import OpenAILLM


def test_initialization() -> None:
    """Test integration initialization."""
    OpenAILLM()
