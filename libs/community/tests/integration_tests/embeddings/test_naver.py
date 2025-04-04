"""Test Naver embeddings."""

from langchain_community.embeddings import ClovaXEmbeddings


def test_embedding_documents() -> None:
    """Test ClovaX embeddings."""
    documents = ["foo bar"]
    embedding = ClovaXEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


async def test_aembedding_documents() -> None:
    """Test ClovaX embeddings."""
    documents = ["foo bar"]
    embedding = ClovaXEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_embedding_query() -> None:
    """Test ClovaX embeddings."""
    document = "foo bar"
    embedding = ClovaXEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0


async def test_aembedding_query() -> None:
    """Test ClovaX embeddings."""
    document = "foo bar"
    embedding = ClovaXEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) > 0
