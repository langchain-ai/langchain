"""Test ZhipuAI Text Embedding."""

from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings


def test_zhipuai_embedding_documents() -> None:
    """Test ZhipuAI Text Embedding for documents."""
    documents = ["This is a test query1.", "This is a test query2."]
    embedding = ZhipuAIEmbeddings()  # type: ignore[call-arg]
    res = embedding.embed_documents(documents)
    assert len(res) == 2
    assert len(res[0]) == 1024


def test_zhipuai_embedding_query() -> None:
    """Test ZhipuAI Text Embedding for query."""
    document = "This is a test query."
    embedding = ZhipuAIEmbeddings()  # type: ignore[call-arg]
    res = embedding.embed_query(document)
    assert len(res) == 1024


def test_zhipuai_embedding_dimensions() -> None:
    """Test ZhipuAI Text Embedding for query by assigning dimensions"""
    document = "This is a test query."
    embedding = ZhipuAIEmbeddings(
        model="embedding-3",
        dimensions=2048,
    )  # type: ignore[call-arg]
    res = embedding.embed_query(document)
    assert len(res) == 2048
