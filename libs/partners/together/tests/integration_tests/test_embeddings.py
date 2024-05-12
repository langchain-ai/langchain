"""Test Together AI embeddings."""

from langchain_together import TogetherEmbeddings


def test_langchain_together_embed_documents() -> None:
    """Test Together AI embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = TogetherEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_together_embed_query() -> None:
    """Test Together AI embeddings."""
    query = "foo bar"
    embedding = TogetherEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_together_aembed_documents() -> None:
    """Test Together AI embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = TogetherEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_together_aembed_query() -> None:
    """Test Together AI embeddings asynchronous."""
    query = "foo bar"
    embedding = TogetherEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0
