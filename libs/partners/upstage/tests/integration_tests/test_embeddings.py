"""Test Upstage embeddings."""
from langchain_upstage import UpstageEmbeddings


def test_langchain_upstage_embed_documents() -> None:
    """Test Upstage embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = UpstageEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_upstage_embed_query() -> None:
    """Test Upstage embeddings."""
    query = "foo bar"
    embedding = UpstageEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_upstage_aembed_documents() -> None:
    """Test Upstage embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = UpstageEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_upstage_aembed_query() -> None:
    """Test Upstage embeddings asynchronous."""
    query = "foo bar"
    embedding = UpstageEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0
