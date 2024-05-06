from unittest.mock import Mock

from langchain_pinecone.vectorstores import Pinecone


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    # mock index
    index = Mock()
    embedding = Mock()
    text_key = "xyz"
    Pinecone(index, embedding, text_key)
