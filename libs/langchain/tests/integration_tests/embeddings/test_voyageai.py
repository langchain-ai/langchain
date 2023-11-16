"""Test voyage embeddings."""
from langchain.embeddings.voyageai import VoyageEmbeddings

# Please set VOYAGE_API_KEY in the environment variables
MODEL = "voyage-01"


def test_voyagi_embedding_documents() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageEmbeddings(model=MODEL)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_voyage_embedding_documents_multiple() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageEmbeddings(model=MODEL, batch_size=2)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_voyage_embedding_query() -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageEmbeddings(model=MODEL)
    output = embedding.embed_query(document)
    assert len(output) == 1024
