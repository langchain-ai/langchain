"""Test Kinetica Chat API wrapper."""
from langchain_kinetica import KineticaLLM


def test_initialization() -> None:
    """Test integration initialization."""
    KineticaLLM()
