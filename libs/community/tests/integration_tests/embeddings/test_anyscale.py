"""Test Anyscale Text Embedding."""
from langchain_community.embeddings.Anyscale import AnyscaleEmbeddings


def test_Anyscale_embedding_documents() -> None:
    """Test Anyscale Text Embedding for documents."""
    documents = ["This is a test", "This is another test"]
    embedding = AnyscaleEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2  # type: ignore[arg-type]
    assert len(output[0]) == 1024  # type: ignore[index]


def test_Anyscale_embedding_query() -> None:
    """Test Anyscale Text Embedding for query."""
    document = "This is a simple test."
    embedding = AnyscaleEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024  # type: ignore[arg-type]
