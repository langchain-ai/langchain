# flake8: noqa
"""Test sentence_transformer embeddings."""

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


def test_sentence_transformer_embedding_documents() -> None:
    """Test sentence_transformer embeddings."""
    embedding = SentenceTransformerEmbeddings()
    documents = ["foo bar"]
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 384


def test_sentence_transformer_embedding_query() -> None:
    """Test sentence_transformer embeddings."""
    embedding = SentenceTransformerEmbeddings()
    query = "what the foo is a bar?"
    query_vector = embedding.embed_query(query)
    assert len(query_vector) == 384


def test_sentence_transformer_db_query() -> None:
    """Test sentence_transformer similarity search."""
    embedding = SentenceTransformerEmbeddings()
    texts = [
        "we will foo your bar until you can't foo any more",
        "the quick brown fox jumped over the lazy dog",
    ]
    query = "what the foo is a bar?"
    query_vector = embedding.embed_query(query)
    assert len(query_vector) == 384
    db = Chroma(embedding_function=embedding)
    db.add_texts(texts)
    docs = db.similarity_search_by_vector(query_vector, k=2)
    assert docs[0].page_content == "we will foo your bar until you can't foo any more"
