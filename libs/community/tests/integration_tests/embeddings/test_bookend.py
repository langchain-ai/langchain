"""Test Bookend AI embeddings."""

from langchain_community.embeddings.bookend import BookendEmbeddings


def test_bookend_embedding_documents() -> None:
    """Test Bookend AI embeddings for documents."""
    documents = ["foo bar", "bar foo"]
    embedding = BookendEmbeddings(
        domain="<bookend_domain>",
        api_token="<bookend_api_token>",
        model_id="<bookend_embeddings_model_id>",
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 768


def test_bookend_embedding_query() -> None:
    """Test Bookend AI embeddings for query."""
    document = "foo bar"
    embedding = BookendEmbeddings(
        domain="<bookend_domain>",
        api_token="<bookend_api_token>",
        model_id="<bookend_embeddings_model_id>",
    )
    output = embedding.embed_query(document)
    assert len(output) == 768
