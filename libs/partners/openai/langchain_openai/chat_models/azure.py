"""Azure OpenAI chat wrapper."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Union

import openai
from langchain_core.outputs import ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_openai.chat_models.base import ChatOpenAI

logger = logging.getLogger(__name__)


class AzureChatOpenAI(ChatOpenAI):
    """`Azure OpenAI` Chat Completion API.

    To use this class you
    must have a deployed model on Azure OpenAI. Use `deployment_name` in the
    constructor to refer to the "Model deployment name" in the Azure portal.

    In addition, you should have the
    following environment variables set or passed in constructor in lower case:
    - ``AZURE_OPENAI_API_KEY``
    - ``AZURE_OPENAI_ENDPOINT``
    - ``AZURE_OPENAI_AD_TOKEN``
    - ``OPENAI_API_VERSION``
    - ``OPENAI_PROXY``

    For example, if you have `gpt-3.5-turbo` deployed, with the deployment name
    `35-turbo-dev`, the constructor should look like:

    .. code-block:: python

        from langchain_openai import AzureChatOpenAI

        AzureChatOpenAI(
            azure_deployment="35-turbo-dev",
            openai_api_version="2023-05-15",
        )

    Be aware the API version may change.

    You can also specify the version of the model using ``model_version`` constructor
    parameter, as Azure OpenAI doesn't return model version with the response.

    Default is empty. When you specify the version, it will be appended to the
    model name in the response. Setting correct version will help you to calculate the
    cost properly. Model version is not validated, so make sure you set it correctly
    to get the correct cost.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.
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
    model_version: str = ""
    """Legacy, for openai<1.0.0 support."""
    openai_api_type: str = ""
    """Legacy, for openai<1.0.0 support."""
    validate_base_url: bool = True
    """For backwards compatibility. If legacy val openai_api_base is passed in, try to 
        infer if it is a base_url or azure_endpoint and update accordingly.
    """

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "azure_openai"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        # Check OPENAI_KEY for backwards compatibility.
        # TODO: Remove OPENAI_API_KEY support to avoid possible conflict when using
        # other forms of azure credentials.
        values["openai_api_key"] = (
            values["openai_api_key"]
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_api_version"] = values["openai_api_version"] or os.getenv(
            "OPENAI_API_VERSION"
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
        )
        values["azure_endpoint"] = values["azure_endpoint"] or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        values["azure_ad_token"] = values["azure_ad_token"] or os.getenv(
            "AZURE_OPENAI_AD_TOKEN"
        )

        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="azure"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values, "openai_proxy", "OPENAI_PROXY", default=""
        )
        # For backwards compatibility. Before openai v1, no distinction was made
        # between azure_endpoint and base_url (openai_api_base).
        openai_api_base = values["openai_api_base"]
        if openai_api_base and values["validate_base_url"]:
            if "/openai" not in openai_api_base:
                raise ValueError(
                    "As of openai>=1.0.0, Azure endpoints should be specified via "
                    "the `azure_endpoint` param not `openai_api_base` "
                    "(or alias `base_url`)."
                )
            if values["deployment_name"]:
                raise ValueError(
                    "As of openai>=1.0.0, if `azure_deployment` (or alias "
                    "`deployment_name`) is specified then "
                    "`base_url` (or alias `openai_api_base`) should not be. "
                    "If specifying `azure_deployment`/`deployment_name` then use "
                    "`azure_endpoint` instead of `base_url`.\n\n"
                    "For example, you could specify:\n\n"
                    'azure_deployment="https://xxx.openai.azure.com/", '
                    'deployment_name="my-deployment"\n\n'
                    "Or you can equivalently specify:\n\n"
                    'base_url="https://xxx.openai.azure.com/openai/deployments/my-deployment"'  # noqa: E501
                )
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
        values["client"] = openai.AzureOpenAI(**client_params).chat.completions
        values["async_client"] = openai.AsyncAzureOpenAI(
            **client_params
        ).chat.completions
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**self._default_params}

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        return {
            "openai_api_type": self.openai_api_type,
            "openai_api_version": self.openai_api_version,
        }

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            if res.get("finish_reason", None) == "content_filter":
                raise ValueError(
                    "Azure has not provided the response due to a content filter "
                    "being triggered"
                )
        chat_result = super()._create_chat_result(response)

        if "model" in response:
            model = response["model"]
            if self.model_version:
                model = f"{model}-{self.model_version}"

            if chat_result.llm_output is not None and isinstance(
                chat_result.llm_output, dict
            ):
                chat_result.llm_output["model_name"] = model

        return chat_result
