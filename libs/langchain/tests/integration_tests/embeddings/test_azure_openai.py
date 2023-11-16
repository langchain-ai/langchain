"""Test openai embeddings."""
import os
from typing import Any

import numpy as np
import pytest

from langchain.embeddings import AzureOpenAIEmbeddings


def _get_embeddings(**kwargs: Any) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        **kwargs,
    )


def test_azure_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = _get_embeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


def test_azure_openai_embedding_documents_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = _get_embeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.asyncio
async def test_azure_openai_embedding_documents_async_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = _get_embeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


def test_azure_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = _get_embeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


@pytest.mark.asyncio
async def test_azure_openai_embedding_async_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = _get_embeddings()
    output = await embedding.aembed_query(document)
    assert len(output) == 1536


@pytest.mark.skip(reason="Unblock scheduled testing. TODO: fix.")
def test_azure_openai_embedding_with_empty_string() -> None:
    """Test openai embeddings with empty string."""
    import openai

    document = ["", "abc"]
    embedding = _get_embeddings()
    output = embedding.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == 1536
    expected_output = openai.Embedding.create(input="", model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == 1536


def test_embed_documents_normalized() -> None:
    output = _get_embeddings().embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


def test_embed_query_normalized() -> None:
    output = _get_embeddings().embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)
