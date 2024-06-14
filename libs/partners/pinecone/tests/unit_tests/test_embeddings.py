"""Test embedding model integration."""

from langchain_core.embeddings import Embeddings

from langchain_pinecone import PineconeEmbeddings

MODEL = "multilingual-e5-large"
MODEL2 = "multilingual-e5-small"


def test_initialization_pinecone_embeddings() -> None:
    """Test embedding model initialization."""
    emb = PineconeEmbeddings(pinecone_api_key="NOT_A_VALID_KEY", model=MODEL)
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == 128
    assert emb.model == MODEL
    assert emb._client is not None


def test_initialization_pinecone_embeddings_batch_size() -> None:
    """Test embedding model initialization."""
    batch_size = 15
    emb = PineconeEmbeddings(
        pinecone_api_key="NOT_A_VALID_KEY", model=MODEL2, batch_size=batch_size
    )
    assert isinstance(emb, Embeddings)
    assert emb.batch_size == batch_size
    assert emb.model == MODEL2
    assert emb._client is not None
