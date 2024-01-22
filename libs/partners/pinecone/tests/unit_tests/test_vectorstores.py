from langchain_pinecone.vectorstores import PineconeVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    PineconeVectorStore()
