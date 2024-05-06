from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import openai
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_openai.llms.base import BaseOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAI(BaseOpenAI):
    """Azure-specific OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_openai import AzureOpenAI

            openai = AzureOpenAI(model_name="gpt-3.5-turbo-instruct")
    """

    azure_endpoint: Union[str, None] = None
    """Your Azure endpoint, including the resource.

        Automatically inferred from env var `AZURE_OPENAI_ENDPOINT` if not provided.

        Example: `https://example-resource.azure.openai.com/`
    """
    deployment_name: Union[str, None] = Field(default=None, alias="azure_deployment")
    """A model deployment. 

        If given sets the base client URL to include `/deployments/{azure_deployment}`.
        Note: this means you won't be able to use non-deployment endpoints.
    """
    openai_api_version: str = Field(default="", alias="api_version")
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Optional[SecretStr] = None
    """Your Azure Active Directory token.

        Automatically inferred from env var `AZURE_OPENAI_AD_TOKEN` if not provided.

        For more: 
        https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-id.
    """  # noqa: E501
    azure_ad_token_provider: Union[Callable[[], str], None] = None
    """A function that returns an Azure Active Directory token.

        Will be invoked on every request.
    """
    openai_api_type: str = ""
    """Legacy, for openai<1.0.0 support."""
    validate_base_url: bool = True
    """For backwards compatibility. If legacy val openai_api_base is passed in, try to 
        infer if it is a base_url or azure_endpoint and update accordingly.
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "openai"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "openai_api_key": "AZURE_OPENAI_API_KEY",
            "azure_ad_token": "AZURE_OPENAI_AD_TOKEN",
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")

        # Check OPENAI_KEY for backwards compatibility.
        # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
        # other forms of azure credentials.
        openai_api_key = (
            values["openai_api_key"]
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        values["openai_api_key"] = (
            convert_to_secret_str(openai_api_key) if openai_api_key else None
        )

        values["azure_endpoint"] = values["azure_endpoint"] or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        azure_ad_token = values["azure_ad_token"] or os.getenv("AZURE_OPENAI_AD_TOKEN")
        values["azure_ad_token"] = (
            convert_to_secret_str(azure_ad_token) if azure_ad_token else None
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["openai_api_version"] = values["openai_api_version"] or os.getenv(
            "OPENAI_API_VERSION"
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = values["openai_api_base"]
        if openai_api_base and values["validate_base_url"]:
            if "/openai" not in openai_api_base:
                values["openai_api_base"] = (
                    values["openai_api_base"].rstrip("/") + "/openai"
                )
                raise ValueError(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`)."
                )
            if values["deployment_name"]:
                raise ValueError(
                    "As of openai>=1.0.0, if `deployment_name` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment_name` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
                values["deployment_name"] = None
        client_params = {
            "api_version": values["openai_api_version"],
            "azure_endpoint": values["azure_endpoint"],
            "azure_deployment": values["deployment_name"],
            "api_key": values["openai_api_key"].get_secret_value()
            if values["openai_api_key"]
            else None,
            "azure_ad_token": values["azure_ad_token"].get_secret_value()
            if values["azure_ad_token"]
            else None,
            "azure_ad_token_provider": values["azure_ad_token_provider"],
            "organization": values["openai_organization"],
            "base_url": values["openai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }
        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.AzureOpenAI(
                **client_params, **sync_specific
            ).completions
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncAzureOpenAI(
                **client_params, **async_specific
            ).completions

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{"deployment_name": self.deployment_name},
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        openai_params = {"model": self.deployment_name}
        return {**openai_params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azure"

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        return {
            "openai_api_type": self.openai_api_type,
            "openai_api_version": self.openai_api_version,
        }
