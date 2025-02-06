"""Test huggingface embeddings."""

import os

import pytest

from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_embedding_documents_on_hpu() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceEmbeddings(model_kwargs={"device": "hpu"})
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_embedding_query_on_hpu() -> None:
    """Test huggingface embeddings."""
    document = "foo bar"
    embedding = HuggingFaceEmbeddings(
        encode_kwargs={"batch_size": 16}, model_kwargs={"device": "hpu"}
    )
    output = embedding.embed_query(document)
    assert len(output) == 768


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_instructor_embedding_documents_on_hpu() -> None:
    """Test huggingface embeddings."""
    documents = ["foo bar"]
    model_name = "hkunlp/instructor-base"
    embedding = HuggingFaceInstructEmbeddings(
        model_name=model_name, model_kwargs={"device": "hpu"}
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_instructor_embedding_query_on_hpu() -> None:
    """Test huggingface embeddings."""
    query = "foo bar"
    model_name = "hkunlp/instructor-base"
    embedding = HuggingFaceInstructEmbeddings(
        model_name=model_name, model_kwargs={"device": "hpu"}
    )
    output = embedding.embed_query(query)
    assert len(output) == 768


@pytest.mark.skipif(not os.getenv("RUN_HPU_TEST"), reason="RUN_HPU_TEST is not set")
def test_huggingface_instructor_embedding_normalize_on_hpu() -> None:
    """Test huggingface embeddings."""
    query = "foo bar"
    model_name = "hkunlp/instructor-base"
    encode_kwargs = {"normalize_embeddings": True}
    embedding = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
        model_kwargs={"device": "hpu"},
    )
    output = embedding.embed_query(query)
    assert len(output) == 768
    eps = 1e-5
    norm = sum([o**2 for o in output])
    assert abs(1 - norm) <= eps
