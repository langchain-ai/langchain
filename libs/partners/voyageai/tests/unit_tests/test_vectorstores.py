from langchain_voyageai.vectorstores import VoyageAIVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    VoyageAIVectorStore()
