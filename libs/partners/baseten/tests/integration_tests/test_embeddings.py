"""Test BasetenEmbeddings integration."""

import os

import pytest

from langchain_baseten import BasetenEmbeddings


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_embedding_model_url")
def test_baseten_embeddings_embed_documents() -> None:
    """Test BasetenEmbeddings embed_documents."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_EMBEDDING_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_EMBEDDING_MODEL_URL not set")

    embeddings = BasetenEmbeddings(
        model="embeddings",
        model_url=model_url,
        baseten_api_key=api_key,
    )

    texts = ["Hello world", "How are you?"]
    result = embeddings.embed_documents(texts)

    assert len(result) == 2
    assert len(result[0]) > 0  # Should have some dimensions
    assert len(result[1]) > 0
    assert isinstance(result[0][0], float)


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_embedding_model_url")
def test_baseten_embeddings_embed_query() -> None:
    """Test BasetenEmbeddings embed_query."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_EMBEDDING_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_EMBEDDING_MODEL_URL not set")

    embeddings = BasetenEmbeddings(
        model="embeddings",
        model_url=model_url,
        baseten_api_key=api_key,
    )

    result = embeddings.embed_query("Hello world")

    assert len(result) > 0  # Should have some dimensions
    assert isinstance(result[0], float)


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_embedding_model_url")
async def test_baseten_embeddings_aembed_documents() -> None:
    """Test BasetenEmbeddings async embed_documents."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_EMBEDDING_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_EMBEDDING_MODEL_URL not set")

    embeddings = BasetenEmbeddings(
        model="embeddings",
        model_url=model_url,
        baseten_api_key=api_key,
    )

    texts = ["Hello world", "How are you?"]
    result = await embeddings.aembed_documents(texts)

    assert len(result) == 2
    assert len(result[0]) > 0  # Should have some dimensions
    assert len(result[1]) > 0
    assert isinstance(result[0][0], float)


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_embedding_model_url")
async def test_baseten_embeddings_aembed_query() -> None:
    """Test BasetenEmbeddings async embed_query."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_EMBEDDING_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_EMBEDDING_MODEL_URL not set")

    embeddings = BasetenEmbeddings(
        model="embeddings",
        model_url=model_url,
        baseten_api_key=api_key,
    )

    result = await embeddings.aembed_query("Hello world")

    assert len(result) > 0  # Should have some dimensions
    assert isinstance(result[0], float)
