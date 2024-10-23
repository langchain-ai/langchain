"""Test Neo4j embeddings."""

from langchain_neo4j.embeddings import Neo4jEmbeddings


def test_langchain_neo4j_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = Neo4jEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_neo4j_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = Neo4jEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
