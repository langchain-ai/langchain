from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import (
    AbstractSet,
    Any,
    AsyncIterator,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import openai
import tiktoken
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env, get_pydantic_field_names
from langchain_core.utils.utils import build_extra_kwargs

from .base import BaseOpenAI

logger = logging.getLogger(__name__)


class AzureOpenAI(BaseOpenAI):
    """Azure-specific OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import AzureOpenAI
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
    openai_api_key: Union[str, None] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `AZURE_OPENAI_API_KEY` if not provided."""
    azure_ad_token: Union[str, None] = None
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
        values["openai_api_key"] = (
            values["openai_api_key"]
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        values["azure_endpoint"] = values["azure_endpoint"] or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        values["azure_ad_token"] = values["azure_ad_token"] or os.getenv(
            "AZURE_OPENAI_AD_TOKEN"
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
                warnings.warn(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    f"the `azure_endpoint` param not `openai_api_base` "
                    f"(or alias `base_url`). Updating `openai_api_base` from "
                    f"{openai_api_base} to {values['openai_api_base']}."
                )
            if values["deployment_name"]:
                warnings.warn(
                    "As of openai>=1.0.0, if `deployment_name` (or alias "
                    "`azure_deployment`) is specified then "
                    "`openai_api_base` (or alias `base_url`) should not be. "
                    "Instead use `deployment_name` (or alias `azure_deployment`) "
                    "and `azure_endpoint`."
                )
                if values["deployment_name"] not in values["openai_api_base"]:
                    warnings.warn(
                        "As of openai>=1.0.0, if `openai_api_base` "
                        "(or alias `base_url`) is specified it is expected to be "
                        "of the form "
                        "https://example-resource.azure.openai.com/openai/deployments/example-deployment. "  # noqa: E501
                        f"Updating {openai_api_base} to "
                        f"{values['openai_api_base']}."
                    )
                    values["openai_api_base"] += (
                        "/deployments/" + values["deployment_name"]
                    )
                values["deployment_name"] = None
        client_params = {
            "api_version": values["openai_api_version"],
            "azure_endpoint": values["azure_endpoint"],
            "azure_deployment": values["deployment_name"],
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
        values["client"] = openai.AzureOpenAI(**client_params).completions
        values["async_client"] = openai.AsyncAzureOpenAI(**client_params).completions

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
