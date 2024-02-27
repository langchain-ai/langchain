"""Test VoyageAI embeddings."""
from langchain_voyageai.embeddings import VoyageAIEmbeddings

MODEL = "voyage-01"


def test_langchain_voyageai_embedding_documents() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar"]
    embedding = VoyageAIEmbeddings(model=MODEL)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_langchain_voyageai_embedding_documents_multiple() -> None:
    """Test voyage embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = VoyageAIEmbeddings(model=MODEL, embed_batch_size=2)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024


def test_langchain_voyageai_embedding_query() -> None:
    """Test voyage embeddings."""
    document = "foo bar"
    embedding = VoyageAIEmbeddings(model=MODEL)
    output = embedding.embed_query(document)
    assert len(output) == 1024
