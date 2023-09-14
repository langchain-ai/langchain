"""Test TensorflowHub embeddings."""
from langchain.embeddings import TensorflowHubEmbeddings


def test_tensorflowhub_embedding_documents() -> None:
    """Test tensorflowhub embeddings."""
    documents = ["foo bar"]
    embedding = TensorflowHubEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 512


def test_tensorflowhub_embedding_query() -> None:
    """Test tensorflowhub embeddings."""
    document = "foo bar"
    embedding = TensorflowHubEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 512
