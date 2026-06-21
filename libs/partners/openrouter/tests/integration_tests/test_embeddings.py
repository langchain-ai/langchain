"""Test OpenRouter Embedding."""

from unittest.mock import patch

import httpx
import pytest
import tenacity

from langchain_openrouter import OpenRouterEmbeddings


def test_openrouter_embedding_documents() -> None:
    """Test OpenRouter embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = OpenRouterEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_openrouter_embedding_query() -> None:
    """Test OpenRouter embeddings for query."""
    document = "foo bar"
    embedding = OpenRouterEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0


async def test_openrouter_embedding_documents_async() -> None:
    """Test OpenRouter embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = OpenRouterEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_openrouter_embedding_documents_tenacity_error_async() -> None:
    """Test OpenRouter embeddings retry behavior."""
    documents = ["foo bar", "test document"]
    embedding = OpenRouterEmbeddings(max_retries=0)
    mock_response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", url=str(embedding.async_client.base_url)),
    )
    with (
        patch.object(embedding.async_client, "post", return_value=mock_response),
        pytest.raises(tenacity.RetryError),
    ):
        await embedding.aembed_documents(documents)


async def test_openrouter_embedding_documents_http_error_async() -> None:
    """Test OpenRouter embeddings error handling."""
    documents = ["foo bar", "test document"]
    embedding = OpenRouterEmbeddings(max_retries=None)
    mock_response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", url=str(embedding.async_client.base_url)),
    )
    with (
        patch.object(embedding.async_client, "post", return_value=mock_response),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await embedding.aembed_documents(documents)


async def test_openrouter_embedding_query_async() -> None:
    """Test OpenRouter embeddings for query."""
    document = "foo bar"
    embedding = OpenRouterEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) > 0


def test_openrouter_embedding_documents_long() -> None:
    """Test OpenRouter embeddings batching."""
    documents = ["foo bar " * 1000, "test document " * 1000] * 5
    embedding = OpenRouterEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 10
    assert len(output[0]) > 0


def test_openrouter_embed_query_unicode() -> None:
    """Test OpenRouter embeddings with unicode."""
    document = "😳"
    embedding = OpenRouterEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
