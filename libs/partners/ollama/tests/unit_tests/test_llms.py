"""Test Ollama Chat API wrapper."""

from langchain_ollama import OllamaLLM


def test_initialization() -> None:
    """Test integration initialization."""
    OllamaLLM(model="llama3")
