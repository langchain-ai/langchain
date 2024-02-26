"""Test Kinetica embeddings."""
from langchain_kinetica.embeddings import KineticaEmbeddings


def test_langchain_kinetica_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = KineticaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_kinetica_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = KineticaEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
