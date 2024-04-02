"""Test embedding model integration."""


from langchain_cloudflare.embeddings import CloudflareEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    CloudflareEmbeddings()
