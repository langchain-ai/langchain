"""Test embeddings base module."""

import pytest

from langchain.embeddings.base import (
    _BUILTIN_PROVIDERS,
    _infer_model_and_provider,
    _parse_model_string,
    get_provider_package,
)


@pytest.mark.parametrize(
    ("model_string", "expected_provider", "expected_model"),
    [
        ("openai:text-embedding-3-small", "openai", "text-embedding-3-small"),
        ("bedrock:amazon.titan-embed-text-v1", "bedrock", "amazon.titan-embed-text-v1"),
        ("huggingface:BAAI/bge-base-en:v1.5", "huggingface", "BAAI/bge-base-en:v1.5"),
        ("google_genai:gemini-embedding-001", "google_genai", "gemini-embedding-001"),
    ],
)
def test_parse_model_string(model_string: str, expected_provider: str, expected_model: str) -> None:
    """Test parsing model strings into provider and model components."""
    assert _parse_model_string(model_string) == (
        expected_provider,
        expected_model,
    )


def test_parse_model_string_errors() -> None:
    """Test error cases for model string parsing."""
    with pytest.raises(ValueError, match="Model name must be"):
        _parse_model_string("just-a-model-name")

    with pytest.raises(ValueError, match="Invalid model format "):
        _parse_model_string("")

    with pytest.raises(ValueError, match="is not supported"):
        _parse_model_string(":model-name")

    with pytest.raises(ValueError, match="Model name cannot be empty"):
        _parse_model_string("openai:")

    with pytest.raises(
        ValueError,
        match="Provider 'invalid-provider' is not supported",
    ):
        _parse_model_string("invalid-provider:model-name")

    for provider in _BUILTIN_PROVIDERS:
        with pytest.raises(ValueError, match=f"{provider}"):
            _parse_model_string("invalid-provider:model-name")


def test_infer_model_and_provider() -> None:
    """Test model and provider inference from different input formats."""
    assert _infer_model_and_provider("openai:text-embedding-3-small") == (
        "openai",
        "text-embedding-3-small",
    )

    assert _infer_model_and_provider(
        model="text-embedding-3-small",
        provider="openai",
    ) == ("openai", "text-embedding-3-small")

    assert _infer_model_and_provider(
        model="ft:text-embedding-3-small",
        provider="openai",
    ) == ("openai", "ft:text-embedding-3-small")

    assert _infer_model_and_provider(model="openai:ft:text-embedding-3-small") == (
        "openai",
        "ft:text-embedding-3-small",
    )


def test_infer_model_and_provider_errors() -> None:
    """Test error cases for model and provider inference."""
    # Test missing provider
    with pytest.raises(ValueError, match="Must specify either"):
        _infer_model_and_provider("text-embedding-3-small")

    # Test empty model
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        _infer_model_and_provider("")

    # Test empty provider with model
    with pytest.raises(ValueError, match="Must specify either"):
        _infer_model_and_provider("model", provider="")

    # Test invalid provider
    with pytest.raises(ValueError, match="Provider 'invalid' is not supported") as exc:
        _infer_model_and_provider("model", provider="invalid")
    # Test provider list is in error
    for provider in _BUILTIN_PROVIDERS:
        assert provider in str(exc.value)


@pytest.mark.parametrize(
    "provider",
    sorted(_BUILTIN_PROVIDERS.keys()),
)
def test_supported_providers_package_names(provider: str) -> None:
    """Test that all supported providers have valid package names."""
    package = _BUILTIN_PROVIDERS[provider][0]
    assert "-" not in package
    assert package.startswith("langchain_")
    assert package.islower()


def test_is_sorted() -> None:
    assert list(_BUILTIN_PROVIDERS) == sorted(_BUILTIN_PROVIDERS.keys())


def test_get_provider_package() -> None:
    """The accessor returns the pip package name for a provider."""
    assert get_provider_package("openai") == "langchain-openai"
    assert get_provider_package("azure_ai") == "langchain-azure-ai"


def test_get_provider_package_matches_registry() -> None:
    """Every provider resolves to a derived name unless a pypi_name is set."""
    for provider, spec in _BUILTIN_PROVIDERS.items():
        expected = spec.pypi_name or spec.module.split(".", 1)[0].replace("_", "-")
        assert get_provider_package(provider) == expected


def test_get_provider_package_unknown() -> None:
    """Unknown providers raise a helpful error."""
    with pytest.raises(ValueError, match="Unsupported provider='bar'"):
        get_provider_package("bar")
