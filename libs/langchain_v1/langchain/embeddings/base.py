"""Factory functions for embeddings."""

import functools
import importlib
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings


def _call(cls: type[Embeddings], **kwargs: Any) -> Embeddings:
    return cls(**kwargs)


_SUPPORTED_PROVIDERS: dict[str, tuple[str, str, Callable[..., Embeddings]]] = {
    "azure_openai": ("langchain_openai", "AzureOpenAIEmbeddings", _call),
    "bedrock": (
        "langchain_aws",
        "BedrockEmbeddings",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "cohere": ("langchain_cohere", "CohereEmbeddings", _call),
    "google_genai": ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", _call),
    "google_vertexai": ("langchain_google_vertexai", "VertexAIEmbeddings", _call),
    "huggingface": (
        "langchain_huggingface",
        "HuggingFaceEmbeddings",
        lambda cls, model, **kwargs: cls(model_name=model, **kwargs),
    ),
    "mistralai": ("langchain_mistralai", "MistralAIEmbeddings", _call),
    "ollama": ("langchain_ollama", "OllamaEmbeddings", _call),
    "openai": ("langchain_openai", "OpenAIEmbeddings", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the embeddings class.
- `class_name`: The name of the embeddings class to import.
- `creator_func`: A callable that instantiates the class with provided kwargs.
"""


@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _get_embeddings_class_creator(provider: str) -> Callable[..., Embeddings]:
    """Return a factory function that creates an embeddings model for the given provider.

    This function is cached to avoid repeated module imports.

    Args:
        provider: The name of the model provider (e.g., `'openai'`, `'cohere'`).

            Must be a key in `_SUPPORTED_PROVIDERS`.

    Returns:
        A callable that accepts model kwargs and returns an `Embeddings` instance for
            the specified provider.

    Raises:
        ValueError: If the provider is not in `_SUPPORTED_PROVIDERS`.
        ImportError: If the provider's integration package is not installed.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        msg = (
            f"Provider '{provider}' is not supported.\n"
            f"Supported providers and their required packages:\n"
            f"{_get_provider_list()}"
        )
        raise ValueError(msg)

    module_name, class_name, creator_func = _SUPPORTED_PROVIDERS[provider]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        pkg = module_name.replace("_", "-")
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg) from e

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


def _get_provider_list() -> str:
    """Get formatted list of providers and their packages."""
    return "\n".join(
        f"  - {p}: {pkg[0].replace('_', '-')}" for p, pkg in _SUPPORTED_PROVIDERS.items()
    )


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
        msg = (
            f"Invalid model format '{model_name}'.\n"
            f"Model name must be in format 'provider:model-name'\n"
            f"Example valid model strings:\n"
            f"  - openai:text-embedding-3-small\n"
            f"  - bedrock:amazon.titan-embed-text-v1\n"
            f"  - cohere:embed-english-v3.0\n"
            f"Supported providers: {_SUPPORTED_PROVIDERS.keys()}"
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
        msg = (
            "Must specify either:\n"
            "1. A model string in format 'provider:model-name'\n"
            "   Example: 'openai:text-embedding-3-small'\n"
            "2. Or explicitly set provider from: "
            f"{_SUPPORTED_PROVIDERS.keys()}"
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


def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Initialize an embedding model from a model name and optional provider.

    !!! note
        Requires the integration package for the chosen model provider to be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        Refer to the [provider integration's API reference](https://docs.langchain.com/oss/python/integrations/providers)
        for supported model parameters to use as `**kwargs`.

    Args:
        model: The name of the model, e.g. `'openai:text-embedding-3-small'`.

            You can also specify model and model provider in a single argument using
            `'{model_provider}:{model}'` format, e.g. `'openai:text-embedding-3-small'`.
        provider: The model provider if not specified as part of the model arg
            (see above).

            Supported `provider` values and the corresponding integration package
            are:

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)

        **kwargs: Additional model-specific parameters passed to the embedding model.

            These vary by provider. Refer to the specific model provider's
            [integration reference](https://reference.langchain.com/python/integrations/)
            for all available parameters.

    Returns:
        An `Embeddings` instance that can generate embeddings for text.

    Raises:
        ValueError: If the model provider is not supported or cannot be determined
        ImportError: If the required provider package is not installed

    ???+ example

        ```python
        # pip install langchain langchain-openai

        # Using a model string
        model = init_embeddings("openai:text-embedding-3-small")
        model.embed_query("Hello, world!")

        # Using explicit provider
        model = init_embeddings(model="text-embedding-3-small", provider="openai")
        model.embed_documents(["Hello, world!", "Goodbye, world!"])

        # With additional parameters
        model = init_embeddings("openai:text-embedding-3-small", api_key="sk-...")
        ```

    !!! version-added "Added in `langchain` 0.3.9"

    """
    if not model:
        providers = _SUPPORTED_PROVIDERS.keys()
        msg = f"Must specify model name. Supported providers are: {', '.join(providers)}"
        raise ValueError(msg)

    provider, model_name = _infer_model_and_provider(model, provider=provider)
    return _get_embeddings_class_creator(provider)(model=model_name, **kwargs)


__all__ = [
    "Embeddings",  # This one is for backwards compatibility
    "init_embeddings",
]
