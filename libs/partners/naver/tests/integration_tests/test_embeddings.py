"""Test Naver embeddings."""
from langchain_naver.embeddings import ClovaStudioEmbeddings


def test_langchain_naver_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = ClovaStudioEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_naver_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = ClovaStudioEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
