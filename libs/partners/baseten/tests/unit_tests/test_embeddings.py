"""Test BasetenEmbeddings."""

import os
import pytest

from langchain_baseten import BasetenEmbeddings


def test_baseten_embeddings_init() -> None:
    """Test BasetenEmbeddings initialization."""
    embeddings = BasetenEmbeddings(
        model="test-model",
        model_url="https://model-123.api.baseten.co/production/predict/sync/v1",
        baseten_api_key="test_key",
    )
    assert embeddings.model == "test-model"
    assert embeddings.model_url == "https://model-123.api.baseten.co/production/predict/sync/v1"


def test_baseten_embeddings_init_missing_api_key() -> None:
    """Test BasetenEmbeddings initialization with missing API key."""
    # Ensure no API key is set in environment
    original_key = os.environ.get("BASETEN_API_KEY")
    if "BASETEN_API_KEY" in os.environ:
        del os.environ["BASETEN_API_KEY"]

    try:
        with pytest.raises(ValueError, match="You must specify an api key"):
            BasetenEmbeddings(
                model="test-model",
                model_url="https://model-123.api.baseten.co/production/predict/sync/v1",
            )
    finally:
        # Restore original key if it existed
        if original_key is not None:
            os.environ["BASETEN_API_KEY"] = original_key


def test_baseten_embeddings_init_missing_model_url() -> None:
    """Test BasetenEmbeddings initialization with missing model URL."""
    with pytest.raises(ValueError, match="Field required"):
        BasetenEmbeddings(
            model="test-model",
            baseten_api_key="test_key",
        )


def test_baseten_embeddings_url_normalization() -> None:
    """Test that model URLs are normalized correctly for Performance Client."""
    # Test /sync/v1 URL gets normalized to /sync for Performance Client
    embeddings1 = BasetenEmbeddings(
        model="test-model",
        model_url="https://model-123.api.baseten.co/production/predict/sync/v1",
        baseten_api_key="test_key",
    )
    # Performance Client should be initialized with /sync URL (without /v1)
    assert embeddings1.client is not None

    # Test /sync URL stays the same
    embeddings2 = BasetenEmbeddings(
        model="test-model",
        model_url="https://model-123.api.baseten.co/production/predict/sync",
        baseten_api_key="test_key",
    )
    assert embeddings2.client is not None


def test_baseten_embeddings_empty_documents() -> None:
    """Test embedding empty list of documents."""
    embeddings = BasetenEmbeddings(
        model="test-model",
        model_url="https://model-123.api.baseten.co/production/predict/sync/v1",
        baseten_api_key="test_key",
    )

    result = embeddings.embed_documents([])
    assert result == []
