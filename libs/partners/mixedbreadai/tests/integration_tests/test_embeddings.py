"""Test MixedbreadAI embeddings."""

from langchain_mixedbreadai.embeddings import MixedbreadAIEmbeddings


def test_langchain_mixedbreadai_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = MixedbreadAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_mixedbreadai_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = MixedbreadAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
