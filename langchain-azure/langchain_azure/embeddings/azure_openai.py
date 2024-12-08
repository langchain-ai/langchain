"""Azure OpenAI embeddings wrapper."""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.utils import get_from_dict_or_env
from pydantic import Field, model_validator
from typing_extensions import Self

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1


@deprecated(
    since="0.0.9",
    removal="1.0",
    alternative_import="langchain_openai.AzureOpenAIEmbeddings",
)
class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    """`Azure OpenAI` Embeddings API."""

    azure_endpoint: Union[str, None] = None
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.
        
        Example: `https://example-resource.azure.openai.com/`
    """
    deployment: Optional[str] = Field(default=None, alias="azure_deployment")
    """A model deployment. 

        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    openai_api_key: Union[str, None] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Union[str, None] = None
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more: 
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every request.
    """
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    validate_base_url: bool = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        # Check OPENAI_KEY for backwards compatibility.
        # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
        # other forms of azure credentials.
        values["openai_api_key"] = (
            values.get("openai_api_key")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        values["openai_api_base"] = values.get("openai_api_base") or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_version"] = values.get("openai_api_version") or os.getenv(
            "OPENAI_API_VERSION", default="2023-05-15"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        values["openai_organization"] = (
            values.get("openai_organization")
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["azure_endpoint"] = values.get("azure_endpoint") or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        values["azure_ad_token"] = values.get("azure_ad_token") or os.getenv(
            "AZURE_OPENAI_AD_TOKEN"
        )
        # Azure OpenAI embedding models allow a maximum of 2048 texts
        # at a time in each batch
        # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        values["chunk_size"] = min(values["chunk_size"], 2048)
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if is_openai_v1():
            # For backwards compatibility. Before openai v1, no distinction was made
            # between azure_endpoint and base_url (openai_api_base).
            openai_api_base = values["openai_api_base"]
            if openai_api_base and values["validate_base_url"]:
                if "/openai" not in openai_api_base:
                    values["openai_api_base"] += "/openai"
                    warnings.warn(
                        "As of openai>=1.0.0, Azure endpoints should be specified via "
                        f"the `azure_endpoint` param not `openai_api_base` "
                        f"(or alias `base_url`). Updating `openai_api_base` from "
                        f"{openai_api_base} to {values['openai_api_base']}."
                    )
                if values["deployment"]:
                    warnings.warn(
                        "As of openai>=1.0.0, if `deployment` (or alias "
                        "`azure_deployment`) is specified then "
                        "`openai_api_base` (or alias `base_url`) should not be. "
                        "Instead use `deployment` (or alias `azure_deployment`) "
                        "and `azure_endpoint`."
                    )
                    if values["deployment"] not in values["openai_api_base"]:
                        warnings.warn(
                            "As of openai>=1.0.0, if `openai_api_base` "
                            "(or alias `base_url`) is specified it is expected to be "
                            "of the form "
                            "https://example-resource.azure.openai.com/openai/deployments/example-deployment. "  # noqa: E501
                            f"Updating {openai_api_base} to "
                            f"{values['openai_api_base']}."
                        )
                        values["openai_api_base"] += (
                            "/deployments/" + values["deployment"]
                        )
                    values["deployment"] = None
        return values

    @model_validator(mode="after")
    def post_init_validator(self) -> Self:
        """Validate that the base url is set."""
        import openai

        if is_openai_v1():
            client_params = {
                "api_version": self.openai_api_version,
                "azure_endpoint": self.azure_endpoint,
                "azure_deployment": self.deployment,
                "api_key": self.openai_api_key,
                "azure_ad_token": self.azure_ad_token,
                "azure_ad_token_provider": self.azure_ad_token_provider,
                "organization": self.openai_organization,
                "base_url": self.openai_api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
                "http_client": self.http_client,
            }
            self.client = openai.AzureOpenAI(**client_params).embeddings
            self.async_client = openai.AsyncAzureOpenAI(**client_params).embeddings
        else:
            self.client = openai.Embedding
        return self

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"