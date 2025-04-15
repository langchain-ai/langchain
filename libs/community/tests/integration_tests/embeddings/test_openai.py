"""Test openai embeddings."""

import numpy as np
import pytest

from langchain_community.embeddings.openai import OpenAIEmbeddings


@pytest.mark.scheduled
def test_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


@pytest.mark.scheduled
def test_openai_embedding_documents_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = OpenAIEmbeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.scheduled
async def test_openai_embedding_documents_async_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = OpenAIEmbeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.scheduled
def test_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


@pytest.mark.scheduled
async def test_openai_embedding_async_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) == 1536


@pytest.mark.skip(reason="Unblock scheduled testing. TODO: fix.")
@pytest.mark.scheduled
def test_openai_embedding_with_empty_string() -> None:
    """Test openai embeddings with empty string."""
    import openai

    document = ["", "abc"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == 1536
    expected_output = openai.Embedding.create(input="", model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == 1536


@pytest.mark.scheduled
def test_embed_documents_normalized() -> None:
    output = OpenAIEmbeddings().embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


@pytest.mark.scheduled
def test_embed_query_normalized() -> None:
    output = OpenAIEmbeddings().embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)
