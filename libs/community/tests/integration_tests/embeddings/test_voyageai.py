"""Test voyage embeddings."""

from langchain_community.embeddings.voyageai import VoyageEmbeddings

# Please set VOYAGE_API_KEY in the environment variables
MODEL = "voyage-2"


def test_voyagi_embedding_documents() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageEmbeddings(model=MODEL)  # type: ignore[call-arg]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_voyagi_with_default_model() -> None:
    """Test voyage embeddings."""
    embedding = VoyageEmbeddings()  # type: ignore[call-arg]
    assert embedding.model == "voyage-01"
    assert embedding.batch_size == 7
    documents = [f"foo bar {i}" for i in range(72)]
    output = embedding.embed_documents(documents)
    assert len(output) == 72
    assert len(output[0]) == 1024


def test_voyage_embedding_documents_multiple() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageEmbeddings(model=MODEL, batch_size=2)
    assert embedding.model == MODEL
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_voyage_embedding_query() -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageEmbeddings(model=MODEL)  # type: ignore[call-arg]
    output = embedding.embed_query(document)
    assert len(output) == 1024
