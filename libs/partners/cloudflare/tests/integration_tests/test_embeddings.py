"""Test Cloudflare embeddings."""
from langchain_cloudflare.embeddings import CloudflareEmbeddings


def test_langchain_cloudflare_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = CloudflareEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_cloudflare_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = CloudflareEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
