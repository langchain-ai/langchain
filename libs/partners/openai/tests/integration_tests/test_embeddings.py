"""Test OpenAI embeddings."""
from langchain_openai.embeddings.base import OpenAIEmbeddings


def test_langchain_openai_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_openai_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
