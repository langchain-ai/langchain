"""Factory functions for embeddings."""

import functools
from importlib import util
from typing import Any

from langchain_core.embeddings import Embeddings

_SUPPORTED_PROVIDERS = {
    "azure_openai": "langchain_openai",
    "bedrock": "langchain_aws",
    "cohere": "langchain_cohere",
    "google_vertexai": "langchain_google_vertexai",
    "huggingface": "langchain_huggingface",
    "mistralai": "langchain_mistralai",
    "ollama": "langchain_ollama",
    "openai": "langchain_openai",
}


def _get_provider_list() -> str:
    """Get formatted list of providers and their packages."""
    return "\n".join(f"  - {p}: {pkg.replace('_', '-')}" for p, pkg in _SUPPORTED_PROVIDERS.items())


def _parse_model_string(model_name: str) -> tuple[str, str]:
    """Parse a model string into provider and model name components.

    The model string should be in the format 'provider:model-name', where provider
    is one of the supported providers.

    Args:
        model_name: A model string in the format 'provider:model-name'

    Returns:
        A tuple of (provider, model_name)

    ```python
    _parse_model_string("openai:text-embedding-3-small")
    # Returns: ("openai", "text-embedding-3-small")

    _parse_model_string("bedrock:amazon.titan-embed-text-v1")
    # Returns: ("bedrock", "amazon.titan-embed-text-v1")
    ```

    Raises:
        ValueError: If the model string is not in the correct format or
            the provider is unsupported

    """
    if ":" not in model_name:
        providers = _SUPPORTED_PROVIDERS
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
            f"Example valid model strings:\n"
            f"  - openai:text-embedding-3-small\n"
            f"  - bedrock:amazon.titan-embed-text-v1\n"
            f"  - cohere:embed-english-v3.0\n"
            f"Supported providers: {providers}"
        )
        raise ValueError(msg)

    provider, model = model_name.split(":", 1)
    provider = provider.lower().strip()
    model = model.strip()

    if provider not in _SUPPORTED_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    if not model:
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    return provider, model


def _infer_model_and_provider(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[str, str]:
    if not model.strip():
        msg = "Model name cannot be empty"
        raise ValueError(msg)
    if provider is None and ":" in model:
        provider, model_name = _parse_model_string(model)
    else:
        model_name = model

    if not provider:
        providers = _SUPPORTED_PROVIDERS
        msg = (
            "Must specify either:\n"
            "1. A model string in format 'provider:model-name'\n"
            "   Example: 'openai:text-embedding-3-small'\n"
            "2. Or explicitly set provider from: "
            f"{providers}"
        )
        raise ValueError(msg)

    if provider not in _SUPPORTED_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)
    return provider, model_name


@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _check_pkg(pkg: str) -> None:
    """Check if a package is installed."""
    if not util.find_spec(pkg):
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg)


def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Initialize an embeddings model from a model name and optional provider.

    !!! note
        Must have the integration package corresponding to the model provider
        installed.

    Args:
        model: Name of the model to use.

            Can be either:

            - A model string like `"openai:text-embedding-3-small"`
            - Just the model name if the provider is specified separately or can be
                inferred.

            See supported providers under the `provider` arg description.
        provider: Optional explicit provider name. If not specified, will attempt to
            parse from the model string in the `model` arg.

            Supported providers:

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `mistraiai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)

        **kwargs: Additional model-specific parameters passed to the embedding model.
            These vary by provider, see the provider-specific documentation for details.

    Returns:
        An `Embeddings` instance that can generate embeddings for text.

    Raises:
        ValueError: If the model provider is not supported or cannot be determined
        ImportError: If the required provider package is not installed

    ???+ note "Example Usage"

        ```python
        # Using a model string
        model = init_embeddings("openai:text-embedding-3-small")
        model.embed_query("Hello, world!")

        # Using explicit provider
        model = init_embeddings(model="text-embedding-3-small", provider="openai")
        model.embed_documents(["Hello, world!", "Goodbye, world!"])

        # With additional parameters
        model = init_embeddings("openai:text-embedding-3-small", api_key="sk-...")
        ```

    !!! version-added "Added in version 0.3.9"

    """
    if not model:
        providers = _SUPPORTED_PROVIDERS.keys()
        msg = f"Must specify model name. Supported providers are: {', '.join(providers)}"
        raise ValueError(msg)

    provider, model_name = _infer_model_and_provider(model, provider=provider)
    pkg = _SUPPORTED_PROVIDERS[provider]
    _check_pkg(pkg)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name, **kwargs)
    if provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(model=model_name, **kwargs)
    if provider == "google_vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(model=model_name, **kwargs)
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=model_name, **kwargs)
    if provider == "cohere":
        from langchain_cohere import CohereEmbeddings

        return CohereEmbeddings(model=model_name, **kwargs)
    if provider == "mistralai":
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=model_name, **kwargs)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name, **kwargs)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name, **kwargs)
    msg = (
        f"Provider '{provider}' is not supported.\n"
        f"Supported providers and their required packages:\n"
        f"{_get_provider_list()}"
    )
    raise ValueError(msg)


__all__ = [
    "Embeddings",  # This one is for backwards compatibility
    "init_embeddings",
]
