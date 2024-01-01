from langchain_ai21.vectorstores import AI21VectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    AI21VectorStore()
