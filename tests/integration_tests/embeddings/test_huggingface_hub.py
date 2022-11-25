"""Test huggingfacehub embeddings."""
from langchain.embeddings import HuggingFaceHubEmbeddings


def test_huggingfacehub_embedding_documents() -> None:
    """Test huggingfacehub embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceHubEmbeddings(task="feature-extraction")
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_huggingfacehub_embedding_query() -> None:
    """Test huggingfacehub embeddings."""
    document = "foo bar"
    embedding = HuggingFaceHubEmbeddings(task="feature-extraction")
    output = embedding.embed_query(document)
    assert len(output) == 768
