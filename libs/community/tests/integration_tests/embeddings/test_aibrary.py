"""Test aibrary embeddings."""

from langchain_community.embeddings.aibrary import (
    AiBraryEmbeddings,
)
import numpy as np


def test_aibrary_embedding_documents() -> None:
    """Test aibrary embeddings."""
    documents = ["foo", "bar"]
    embedding = AiBraryEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1536


def test_aibrary_embedding_query() -> None:
    """Test aibrary embeddings."""
    document = "foo bar"
    embedding = AiBraryEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


def test_aibrary_embed_documents_normalized() -> None:
    output = AiBraryEmbeddings().embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


def test_aibrary_embed_query_normalized() -> None:
    output = AiBraryEmbeddings().embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)
