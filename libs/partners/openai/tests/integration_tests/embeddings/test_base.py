"""Test OpenAI embeddings."""
from langchain_openai.embeddings.base import OpenAIEmbeddings


def test_langchain_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0


def test_langchain_openai_embeddings_dimensions() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 128
