"""Test NVAIPlay Chat API wrapper."""
from langchain_nvidia_aiplay import NVAIPlayLLM


def test_integration_initialization() -> None:
    """Test integration initialization."""
    NVAIPlayLLM()
