from langchain_nomic.vectorstores import NomicVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    NomicVectorStore()
