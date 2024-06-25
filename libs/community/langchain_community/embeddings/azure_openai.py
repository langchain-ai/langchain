"""Azure OpenAI embeddings wrapper."""

from __future__ import annotations

import os
import warnings
from typing import Callable, Dict, Optional, Union

from langchain_core._api.deprecation import deprecated
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1


@deprecated(
    since="0.0.9",
    removal="0.3.0",
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
    """Your Azure Active Directory token to access Azure OpenAI Service.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.
        To get an access token, make sure current login user have at least 
        `Cognitive Services User` role, and run the following command:
        .. code-block:: bash

            az account get-access-token --resource https://cognitiveservices.azure.com --query "accessToken" -o tsv

        For more:
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """  # noqa: E501
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every request.
    """
    openai_api_version: Optional[str] = Field(default=None, alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    validate_base_url: bool = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "AZURE_OPENAI_API_KEY"
        )
        values["openai_api_key"] = (
            convert_to_secret_str(openai_api_key) if openai_api_key else None
        )
        values["openai_api_version"] = get_from_dict_or_env(
            values,"openai_api_version","OPENAI_API_VERSION"
        )
        values["azure_endpoint"] = get_from_dict_or_env(
            values, "azure_endpoint", "AZURE_OPENAI_ENDPOINT"
        )

        values["openai_api_base"] = get_from_dict_or_env(
            values, "openai_api_base", "OPENAI_API_BASE"
        )

        azure_ad_token = get_from_dict_or_env(
            values, "azure_ad_token", "AZURE_OPENAI_AD_TOKEN")
        values["azure_ad_token"] = (
            convert_to_secret_str(azure_ad_token) if azure_ad_token else None
        )

        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )

        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values, "openai_proxy", "OPENAI_PROXY", default=""
        )
        # Azure OpenAI embedding models allow a maximum of 16 texts
        # at a time in each batch
        # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
        values["chunk_size"] = min(values["chunk_size"], 16)
        try:
            import openai

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
            client_params = {
                "api_version": values["openai_api_version"],
                "azure_endpoint": values["azure_endpoint"],
                "azure_deployment": values["deployment"],
                "api_key": values["openai_api_key"],
                "azure_ad_token": values["azure_ad_token"],
                "azure_ad_token_provider": values["azure_ad_token_provider"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
                "default_headers": values["default_headers"],
                "default_query": values["default_query"],
                "http_client": values["http_client"],
            }
            values["client"] = openai.AzureOpenAI(**client_params).embeddings
            values["async_client"] = openai.AsyncAzureOpenAI(**client_params).embeddings
        else:
            values["client"] = openai.Embedding
        return values

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"
