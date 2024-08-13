"""Test Databricks embeddings."""

from langchain_databricks.embeddings import DatabricksEmbeddings


def test_langchain_databricks_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = DatabricksEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_databricks_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = DatabricksEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
