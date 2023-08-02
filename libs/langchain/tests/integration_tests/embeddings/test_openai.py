"""Test openai embeddings."""
import numpy as np
import openai
import pytest

from langchain.embeddings.openai import OpenAIEmbeddings


def test_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


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


@pytest.mark.asyncio
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


def test_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


@pytest.mark.asyncio
async def test_openai_embedding_async_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = OpenAIEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) == 1536


def test_openai_embedding_with_empty_string() -> None:
    """Test openai embeddings with empty string."""
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


@pytest.mark.asyncio
async def test_sync_async_embed_documents_equal() -> None:
    documents = ["foo bar", "bar foo", "foo"]
    embedding = OpenAIEmbeddings()
    sync_output = embedding.embed_documents(documents)
    async_output = await embedding.aembed_documents(documents)
    assert np.isclose(sync_output, async_output, atol=1e-4).sum() > 0.99


@pytest.mark.asyncio
async def test_sync_async_embed_query_equal() -> None:
    query = "foo bar"
    embedding = OpenAIEmbeddings()
    sync_output = embedding.embed_query(query)
    async_output = await embedding.aembed_query(query)
    assert np.isclose(sync_output, async_output, atol=1e-4).sum() > 0.99


@pytest.mark.asyncio
async def test_with_without_tokenizing_equal() -> None:
    documents = ["foo bar baz bum buzz boom"]
    with_tokenizing = OpenAIEmbeddings()
    without_tokenizing = OpenAIEmbeddings(embedding_ctx_length=None)
    tokenize_output = with_tokenizing.embed_documents(documents)
    text_output = without_tokenizing.embed_documents(documents)
    assert np.isclose(tokenize_output, text_output, atol=1e-4).sum() > 0.99

    atokenize_output = await with_tokenizing.aembed_documents(documents)
    atext_output = await without_tokenizing.aembed_documents(documents)
    assert np.isclose(atokenize_output, atext_output, atol=1e-4).sum() > 0.99
