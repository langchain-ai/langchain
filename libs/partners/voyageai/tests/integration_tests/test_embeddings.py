"""Test VoyageAI embeddings."""

from langchain_voyageai import VoyageAIEmbeddings

# Please set VOYAGE_API_KEY in the environment variables
MODEL = "voyage-2"


def test_langchain_voyageai_embedding_documents() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model=MODEL)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_langchain_voyageai_embedding_documents_multiple() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIEmbeddings(model=MODEL, batch_size=2)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_langchain_voyageai_embedding_query() -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings(model=MODEL)  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1024


async def test_langchain_voyageai_async_embedding_documents_multiple() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIEmbeddings(model=MODEL, batch_size=2)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


async def test_langchain_voyageai_async_embedding_query() -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings(model=MODEL)  # type: ignore[call-arg]
    output = await embedding.aembed_query(document)
    assert len(output) == 1024


def test_langchain_voyageai_embedding_documents_with_output_dimension() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model="voyage-3-large", output_dimension=256)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256
