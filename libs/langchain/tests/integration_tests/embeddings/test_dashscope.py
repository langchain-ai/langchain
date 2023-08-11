"""Test dashscope embeddings."""
import numpy as np

from langchain.embeddings.dashscope import DashScopeEmbeddings


def test_dashscope_embedding_documents() -> None:
    """Test dashscope embeddings."""
    documents = ["foo bar"]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


def test_dashscope_embedding_documents_multiple() -> None:
    """Test dashscope embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


def test_dashscope_embedding_query() -> None:
    """Test dashscope embeddings."""
    document = "foo bar"
    embedding = DashScopeEmbeddings(model="text-embedding-v1")
    output = embedding.embed_query(document)
    assert len(output) == 1536


def test_dashscope_embedding_with_empty_string() -> None:
    """Test dashscope embeddings with empty string."""
    import dashscope

    document = ["", "abc"]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")
    output = embedding.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == 1536
    expected_output = dashscope.TextEmbedding.call(
        input="", model="text-embedding-v1", text_type="document"
    ).output["embeddings"][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == 1536


if __name__ == "__main__":
    test_dashscope_embedding_documents()
    test_dashscope_embedding_documents_multiple()
    test_dashscope_embedding_query()
    test_dashscope_embedding_with_empty_string()
