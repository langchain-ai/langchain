"""Test MistralAI Embedding."""

from unittest.mock import patch

import httpx
import pytest

from langchain_mistralai import MistralAIEmbeddings


def test_mistralai_embedding_documents() -> None:
    """Test MistralAI embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = MistralAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


def test_mistralai_embedding_query() -> None:
    """Test MistralAI embeddings for query."""
    document = "foo bar"
    embedding = MistralAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024


async def test_mistralai_embedding_documents_async() -> None:
    """Test MistralAI embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = MistralAIEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024


async def test_mistralai_embedding_documents_http_error_async() -> None:
    """Test MistralAI embeddings for documents."""
    documents = ["foo bar", "test document"]
    embedding = MistralAIEmbeddings()
    mock_response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", url=embedding.async_client.base_url),
    )
    with (
        patch.object(embedding.async_client, "post", return_value=mock_response),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await embedding.aembed_documents(documents)


async def test_mistralai_embedding_query_async() -> None:
    """Test MistralAI embeddings for query."""
    document = "foo bar"
    embedding = MistralAIEmbeddings()
    output = await embedding.aembed_query(document)
    assert len(output) == 1024


def test_mistralai_embedding_documents_long() -> None:
    """Test MistralAI embeddings for documents."""
    documents = ["foo bar " * 1000, "test document " * 1000] * 5
    embedding = MistralAIEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 10
    assert len(output[0]) == 1024


def test_mistralai_embed_query_character() -> None:
    """Test MistralAI embeddings for query."""
    document = "😳"
    embedding = MistralAIEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1024
