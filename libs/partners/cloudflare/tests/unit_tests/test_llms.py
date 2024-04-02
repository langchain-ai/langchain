"""Test Cloudflare Chat API wrapper."""
from langchain_cloudflare import CloudflareLLM


def test_initialization() -> None:
    """Test integration initialization."""
    CloudflareLLM()
