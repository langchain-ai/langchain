"""Test HuggingFaceHub embeddings."""

from langchain_community.embeddings import HuggingFaceHubEmbeddings


def test_huggingfacehub_embedding_documents_on_hpu() -> None:
    """Test huggingfacehub embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceHubEmbeddings(model_kwargs={"device": "hpu"})  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


async def test_huggingfacehub_embedding_async_documents_on_hpu() -> None:
    """Test huggingfacehub embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceHubEmbeddings(model_kwargs={"device": "hpu"})  # type: ignore[call-arg]
    output = await embedding.aembed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_huggingfacehub_embedding_query_on_hpu() -> None:
    """Test huggingfacehub embeddings."""
    document = "foo bar"
    embedding = HuggingFaceHubEmbeddings(model_kwargs={"device": "hpu"})  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 768


async def test_huggingfacehub_embedding_async_query_on_hpu() -> None:
    """Test huggingfacehub embeddings."""
    document = "foo bar"
    embedding = HuggingFaceHubEmbeddings(model_kwargs={"device": "hpu"})  # type: ignore[call-arg]
    output = await embedding.aembed_query(document)
    assert len(output) == 768
