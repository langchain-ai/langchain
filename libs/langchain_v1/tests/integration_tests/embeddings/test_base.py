"""Test embeddings base module."""

import importlib

import pytest
from langchain_core.embeddings import Embeddings

from langchain.embeddings.base import _SUPPORTED_PROVIDERS, init_embeddings


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "text-embedding-3-large"),
        ("google_vertexai", "text-embedding-gecko@003"),
        ("bedrock", "amazon.titan-embed-text-v1"),
        ("cohere", "embed-english-v2.0"),
    ],
)
async def test_init_embedding_model(provider: str, model: str) -> None:
    package = _SUPPORTED_PROVIDERS[provider][0]
    try:
        importlib.import_module(package)
    except ImportError:
        pytest.skip(f"Package {package} is not installed")

    model_colon = init_embeddings(f"{provider}:{model}")
    assert isinstance(model_colon, Embeddings)

    model_explicit = init_embeddings(
        model=model,
        provider=provider,
    )
    assert isinstance(model_explicit, Embeddings)

    text = "Hello world"

    embedding_colon = await model_colon.aembed_query(text)
    assert isinstance(embedding_colon, list)
    assert all(isinstance(x, float) for x in embedding_colon)

    embedding_explicit = await model_explicit.aembed_query(text)
    assert isinstance(embedding_explicit, list)
    assert all(isinstance(x, float) for x in embedding_explicit)
