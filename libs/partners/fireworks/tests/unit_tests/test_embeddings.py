"""Test embedding model integration."""

from pydantic import SecretStr

from langchain_fireworks.embeddings import FireworksEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    FireworksEmbeddings(
        model="nomic-ai/nomic-embed-text-v1.5",
        api_key=SecretStr("test_api_key"),
    )
