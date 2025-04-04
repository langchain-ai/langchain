"""Test openai embeddings."""

import os
from typing import Any

import numpy as np
import pytest

from langchain_community.embeddings import AzureOpenAIEmbeddings

OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "")
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", ""),
)


def _get_embeddings(**kwargs: Any) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(  # type: ignore[call-arg]
        azure_deployment=DEPLOYMENT_NAME,
        api_version=OPENAI_API_VERSION,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        **kwargs,
    )


@pytest.mark.scheduled
def test_azure_openai_embedding_documents() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = _get_embeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


@pytest.mark.scheduled
def test_azure_openai_embedding_documents_multiple() -> None:
    """Test openai embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = _get_embeddings(chunk_size=2)
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    assert embedding.chunk_size == 2
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.scheduled
def test_azure_openai_embedding_documents_chunk_size() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"] * 20
    embedding = _get_embeddings()
    embedding.embedding_ctx_length = 8191
    output = embedding.embed_documents(documents)
    # Max 16 chunks per batch on Azure OpenAI embeddings
    assert embedding.chunk_size == 16
    assert len(output) == 20
    assert all([len(out) == 1536 for out in output])


@pytest.mark.scheduled
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


@pytest.mark.scheduled
def test_azure_openai_embedding_query() -> None:
    """Test openai embeddings."""
    document = "foo bar"
    embedding = _get_embeddings()
    output = embedding.embed_query(document)
    assert len(output) == 1536


@pytest.mark.scheduled
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
    expected_output = openai.Embedding.create(input="", model="text-embedding-ada-002")[  # type: ignore[attr-defined]
        "data"
    ][0]["embedding"]
    assert np.allclose(output[0], expected_output)
    assert len(output[1]) == 1536


@pytest.mark.scheduled
def test_embed_documents_normalized() -> None:
    output = _get_embeddings().embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


@pytest.mark.scheduled
def test_embed_query_normalized() -> None:
    output = _get_embeddings().embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)
