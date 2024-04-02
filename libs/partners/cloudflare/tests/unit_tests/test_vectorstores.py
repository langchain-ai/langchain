from langchain_cloudflare.vectorstores import CloudflareVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    CloudflareVectorStore()
