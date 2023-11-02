"""Test Bookend AI embeddings."""
from langchain.embeddings.bookend import BookendEmbeddings


def test_bookend_embedding_documents(
    domain: str, api_token: str, model_id: str
) -> None:
    """Test Bookend AI embeddings for documents."""
    documents = ["foo bar", "bar foo"]

    embedding = BookendEmbeddings(domain=domain, api_token=api_token, model_id=model_id)
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 768


def test_bookend_embedding_query(
        domain: str, api_token: str, model_id: str
) -> None:
    """Test Bookend AI embeddings for query."""
    document = "foo bar"
    embedding = BookendEmbeddings(domain=domain, api_token=api_token, model_id=model_id)
    output = embedding.embed_query(document)
    assert len(output) == 768
