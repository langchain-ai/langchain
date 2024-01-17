"""Test MistralAI Embedding"""
from langchain_mistralai import MistralAIEmbeddings


def test_mistralai_embedding_documents() -> None:
    """Test MistralAI embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = MistralAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_mistralai_embedding_query() -> None:
    """Test MistralAI embeddings for query."""
    document = "foo bar"
    embedding = MistralAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024
