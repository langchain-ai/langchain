"""Test huggingface embeddings."""
import unittest

from langchain.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)


@unittest.skip("This test causes a segfault.")
def test_huggingface_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


@unittest.skip("This test causes a segfault.")
def test_huggingface_embedding_query() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    embedding = HuggingFaceEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_huggingface_instructor_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceInstructEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_huggingface_instructor_embedding_query() -> None:
    """Test huggingface embeddings."""
    query = "foo bar"
    embedding = HuggingFaceInstructEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) == 768
