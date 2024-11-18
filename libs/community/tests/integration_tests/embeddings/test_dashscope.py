"""Test dashscope embeddings."""

import numpy as np

from langchain_community.embeddings.dashscope import DashScopeEmbeddings


def test_dashscope_embedding_documents() -> None:
    """Test dashscope embeddings."""
    documents = ["foo bar"]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


def test_dashscope_embedding_documents_multiple() -> None:
    """Test dashscope embeddings."""
    documents = [
        "foo bar",
        "bar foo",
        "foo",
        "foo0",
        "foo1",
        "foo2",
        "foo3",
        "foo4",
        "foo5",
        "foo6",
        "foo7",
        "foo8",
        "foo9",
        "foo10",
        "foo11",
        "foo12",
        "foo13",
        "foo14",
        "foo15",
        "foo16",
        "foo17",
        "foo18",
        "foo19",
        "foo20",
        "foo21",
        "foo22",
        "foo23",
        "foo24",
    ]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 28
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


def test_dashscope_embedding_query() -> None:
    """Test dashscope embeddings."""
    document = "foo bar"
    embedding = DashScopeEmbeddings(model="text-embedding-v1")  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1536


def test_dashscope_embedding_with_empty_string() -> None:
    """Test dashscope embeddings with empty string."""
    import dashscope

    document = ["", "abc"]
    embedding = DashScopeEmbeddings(model="text-embedding-v1")  # type: ignore[call-arg]
    output = embedding.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == 1536
    expected_output = dashscope.TextEmbedding.call(
        input="", model="text-embedding-v1", text_type="document"
    ).output["embeddings"][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == 1536


def test_dashscope_embedding_with_alias() -> None:
    """Test dashscope embeddings with `api_key` alias."""
    api_key = "your-api-key"
    embeddings = DashScopeEmbeddings(api_key=api_key)
    assert embeddings.dashscope_api_key == api_key


if __name__ == "__main__":
    test_dashscope_embedding_documents()
    test_dashscope_embedding_documents_multiple()
    test_dashscope_embedding_query()
    test_dashscope_embedding_with_empty_string()
