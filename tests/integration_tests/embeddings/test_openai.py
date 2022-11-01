"""Test openai embeddings."""
from langchain.embeddings.openai import OpenAIEmbeddings


def test_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 2048


def test_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 2048
