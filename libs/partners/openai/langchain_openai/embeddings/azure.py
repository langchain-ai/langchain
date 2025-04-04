"""Azure OpenAI embeddings wrapper."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional, Union

import openai
from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self, cast

from langchain_openai.embeddings.base import OpenAIEmbeddings


class AzureOpenAIEmbeddings(OpenAIEmbeddings):  # type: ignore[override]
    """AzureOpenAI embedding model integration.

    Setup:
        To access AzureOpenAI embedding models you'll need to create an Azure account,
        get an API key, and install the `langchain-openai` integration package.

        You’ll need to have an Azure OpenAI instance deployed.
        You can deploy a version on Azure Portal following this
        [guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal).

        Once you have your instance running, make sure you have the name of your
        instance and key. You can find the key in the Azure Portal,
        under the “Keys and Endpoint” section of your instance.

        .. code-block:: bash

            pip install -U langchain_openai

            # Set up your environment variables (or pass them directly to the model)
            export AZURE_OPENAI_API_KEY="your-api-key"
            export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
            export AZURE_OPENAI_API_VERSION="2024-02-01"

    Key init args — completion params:
        model: str
            Name of AzureOpenAI model to use.
        dimensions: Optional[int]
            Number of dimensions for the embeddings. Can be specified only
            if the underlying model supports it.

    Key init args — client params:
      api_key: Optional[SecretStr]

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_openai import AzureOpenAIEmbeddings

            embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-large"
                # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
                # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
                # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
                # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """  # noqa: E501

    azure_endpoint: Optional[str] = Field(
        default_factory=from_env("AZURE_OPENAI_ENDPOINT", default=None)
    )
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment: Optional[str] = Field(default=None, alias="azure_deployment")
    """A model deployment.

        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    # Check OPENAI_KEY for backwards compatibility.
    # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
    # other forms of azure credentials.
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], default=None
        ),
    )
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    openai_api_version: Optional[str] = Field(
        default_factory=from_env("OPENAI_API_VERSION", default="2023-05-15"),
        alias="api_version",
    )
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided.
    
    Set to "2023-05-15" by default if env variable `OPENAI_API_VERSION` is not set.
    """
    azure_ad_token: Optional[SecretStr] = Field(
        default_factory=secret_from_env("AZURE_OPENAI_AD_TOKEN", default=None)
    )
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more:
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every sync request. For async requests,
        will be invoked if `azure_ad_async_token_provider` is not provided.
    """
    azure_ad_async_token_provider: Union[Callable[[], Awaitable[str]], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every async request.
    """
    openai_api_type: Optional[str] = Field(
        default_factory=from_env("OPENAI_API_TYPE", default="azure")
    )
    validate_base_url: bool = True
    chunk_size: int = 2048
    """Maximum number of texts to embed in each batch"""

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = self.openai_api_base
        if openai_api_base and self.validate_base_url:
            if "/openai" not in openai_api_base:
                self.openai_api_base = cast(str, self.openai_api_base) + "/openai"
                raise ValueError(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`). "
                )
            if self.deployment:
                raise ValueError(
                    "As of openai>=1.0.0, if `deployment` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
        client_params: dict = {
            "api_version": self.openai_api_version,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.deployment,
            "api_key": (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            ),
            "azure_ad_token": (
                self.azure_ad_token.get_secret_value() if self.azure_ad_token else None
            ),
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": {
                **(self.default_headers or {}),
                "User-Agent": "langchain-partner-python-azure-openai",
            },
            "default_query": self.default_query,
        }
        if not self.client:
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.AzureOpenAI(
                **client_params,  # type: ignore[arg-type]
                **sync_specific,
            ).embeddings
        if not self.async_client:
            async_specific: dict = {"http_client": self.http_async_client}

            if self.azure_ad_async_token_provider:
                client_params["azure_ad_token_provider"] = (
                    self.azure_ad_async_token_provider
                )

            self.async_client = openai.AsyncAzureOpenAI(
                **client_params,  # type: ignore[arg-type]
                **async_specific,
            ).embeddings
        return self

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"
