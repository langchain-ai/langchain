"""Test optimized huggingface embeddings."""

from langchain_community.embeddings.intel_optimized_embeddings import (
    OptimizedHuggingFaceEmbeddings,
    OptimizedHuggingFaceBgeEmbeddings,
    OptimizedHuggingFaceInstructEmbeddings,
)

def test_optimized_huggingface_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = OptimizedHuggingFaceEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_optimized_huggingface_embedding_query() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    embedding = OptimizedHuggingFaceEmbeddings(encode_kwargs={"batch_size": 16})
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_optimized_huggingface_bge_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = OptimizedHuggingFaceBgeEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


def test_optimized_huggingface_bge_embedding_query() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    embedding = OptimizedHuggingFaceBgeEmbeddings(encode_kwargs={"batch_size": 16})
    output = embedding.embed_query(document)
    assert len(output) == 1024


def test_optimized_huggingface_instructor_embedding_documents() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = OptimizedHuggingFaceInstructEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_optimized_huggingface_instructor_embedding_query() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    embedding = OptimizedHuggingFaceInstructEmbeddings(encode_kwargs={"batch_size": 16})
    output = embedding.embed_query(document)
    assert len(output) == 768
