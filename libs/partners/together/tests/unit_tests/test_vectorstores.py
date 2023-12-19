from langchain_together.vectorstores import TogetherVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    TogetherVectorStore()
