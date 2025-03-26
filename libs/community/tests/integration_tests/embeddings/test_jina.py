"""Test jina embeddings."""

from langchain_community.embeddings.jina import JinaEmbeddings, JinaLateChunkEmbeddings


def test_jina_embedding_documents() -> None:
    """Test jina embeddings for documents."""
    documents = ["foo bar", "bar foo"]
    embedding = JinaEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 512


def test_jina_embedding_query() -> None:
    """Test jina embeddings for query."""
    document = "foo bar"
    embedding = JinaEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 512
    
    
def test_jina_late_chunk_embedding_documents() -> None:
    """Test jina embeddings for documents."""
    documents = ["foo bar", "bar foo"]
    embedding = JinaLateChunkEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_jina_late_chunk_embedding_query() -> None:
    """Test jina embeddings for query."""
    document = "foo bar"
    embedding = JinaLateChunkEmbeddings()  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1024
