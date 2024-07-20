"""Test Ollama embeddings."""

from langchain_ollama.embeddings import OllamaEmbeddings


def test_langchain_ollama_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = OllamaEmbeddings(model="llama3")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_ollama_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = OllamaEmbeddings(model="llama3")
    output = embedding.embed_query(document)
    assert len(output) > 0
