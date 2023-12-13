"""Test NVAIPlay Chat API wrapper."""
from nvidia_aiplay import NVAIPlayLLM


def test_integration_initialization() -> None:
    """Test integration initialization."""
    NVAIPlayLLM()
