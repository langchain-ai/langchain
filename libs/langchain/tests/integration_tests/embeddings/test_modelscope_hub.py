"""Test modelscope embeddings."""
from langchain.embeddings.modelscope_hub import ModelScopeEmbeddings


def test_modelscope_embedding_documents() -> None:
    """Test modelscope embeddings for documents."""
    documents = ["foo bar"]
    embedding = ModelScopeEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 512


def test_modelscope_embedding_query() -> None:
    """Test modelscope embeddings for query."""
    document = "foo bar"
    embedding = ModelScopeEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 512
