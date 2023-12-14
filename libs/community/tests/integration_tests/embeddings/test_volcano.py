"""Test Bytedance Vocalno Embedding."""
from langchain_community.embeddings import VolcanoEmbeddings


def test_modelscope_embedding_documents() -> None:
    """Test modelscope embeddings for documents."""
    documents = ["foo", "bar"]
    embedding = VolcanoEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_modelscope_embedding_query() -> None:
    """Test modelscope embeddings for query."""
    document = "foo bar"
    embedding = VolcanoEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024
