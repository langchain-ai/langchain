from langchain_cohere.vectorstores import CohereVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    CohereVectorStore()
