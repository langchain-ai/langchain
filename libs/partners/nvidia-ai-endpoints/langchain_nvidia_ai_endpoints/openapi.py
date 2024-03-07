from __future__ import annotations

import os
from typing import Any, Callable, Union, Mapping
from typing_extensions import Self, override

import httpx

from openai import OpenAI as SyncOpenAI, AsyncOpenAI
from openai import resources
from openai._types import NOT_GIVEN, Omit, Timeout

from openai._base_client import (
    SyncAPIClient,
    AsyncAPIClient,
    DEFAULT_MAX_RETRIES,
)

from langchain_core.pydantic_v1 import Field, root_validator
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings, OpenAIEmbeddings


class ClientMixin:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        use_base_as_endpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct a new synchronous openai client instance."""
        if api_key is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
        if api_key is None:
            raise Exception(
                "The api_key client option must be set either by passing api_key"
                " to the client or by setting the NVIDIA_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("NVIDIA_BASE_URL")
        if base_url is None:
            base_url = f"https://api.nvidia.com/v1"
        
        self.use_base_as_endpoint = use_base_as_endpoint

        super().__init__(
            api_key = api_key,
            organization = organization,
            base_url = base_url,
            timeout = timeout,
            max_retries = max_retries,
            default_headers = default_headers,
            default_query = default_query,
            http_client = http_client,
            _strict_response_validation = _strict_response_validation,
        )

    def default_headers(self) -> dict[str, str | Omit]:
        return {**super().default_headers, **self._custom_headers}

    def _prepare_url(self, url: str) -> httpx.URL:
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        merge_url = httpx.URL(url)
        if self.use_base_as_endpoint:
            merge_url = httpx.URL(str(self.base_url).rstrip("/"))
        elif merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b"/")
            merge_url = self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url


class SyncNVIDIA(ClientMixin, SyncOpenAI):
    pass


class AsyncNVIDIA(ClientMixin, AsyncOpenAI):
    pass


class NVIDIAMixin:

    api_key: Optional[str] = Field(default=None, alias="nvidia_api_key")
    """Automatically inferred from env var `NVIDIA_API_KEY` if not provided."""
    base_url: Optional[str] = Field(
        default="https://integrate.api.nvidia.com/v1", 
        alias="nvidia_api_base"
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    organization: Optional[str] = Field(default=None, alias="nvidia_organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""    
    # to support explicit proxy for OpenAI
    proxy: Optional[str] = Field(default=None, alias="nvidia_proxy")
    api_version: Optional[str] = Field(default=None, alias="nvidia_api_version")
    api_type: Optional[str] = Field(default=None, alias="nvidia_api_type")
    partner_name = "NVIDIA"

    class Config:
        fields = {
            "openai_api_key": {"exclude": True},
            "openai_api_type": {"exclude": True},
            "openai_api_version": {"exclude": True},
            "openai_organization": {"exclude": True},
            "openai_api_base": {"exclude": True},
        }

    @root_validator(pre=True)
    def get_environment(cls, values: dict) -> dict:
        env_key = os.getenv(f"{cls.partner_name}_API_KEY")
        env_url1 = os.getenv(f"{cls.partner_name}_API_BASE")
        env_url2 = os.getenv(f"{cls.partner_name}_BASE_URL")
        env_org1 = os.getenv(f"{cls.partner_name}_ORG_ID")
        env_org2 = os.getenv(f"{cls.partner_name}_ORGANIZATION")
        env_prox = os.getenv(f"{cls.partner_name}_PROXY")
        env_ver1 = os.getenv(f"{cls.partner_name}_API_VERSION")
        env_ver2 = os.getenv(f"{cls.partner_name}_VERSION")
        env_type = os.getenv(f"{cls.partner_name}_API_TYPE")

        api_key = values.get("api_key") or env_key
        values["api_key"] = (convert_to_secret_str(api_key) if api_key else None)
        values["base_url"] = values["base_url"] or env_url1 or env_url2
        values["proxy"] = values.get("proxy") or env_prox or ""
        values["api_version"] = values.get("api_version") or env_ver1 or env_ver2
        values["env_type"] = values.get("env_type") or env_type

        return values
    
    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.completion
    
    @staticmethod
    def _set_client(values: dict, client_fn: Callable):
        client_params = {
            "api_key": (
                values["api_key"].get_secret_value()
                if values["api_key"]
                else None
            ),
            "organization": values["organization"],
            "base_url": values["base_url"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
            "http_client": values["http_client"],
        }

        if not values.get("client"):
            values["client"] = _get_resource(SyncNVIDIA(**client_params))
        if not values.get("async_client"):
            values["async_client"] = _get_resource(AyncNVIDIA(**client_params))
        return values
    
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "NVIDIA_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.openai_api_base:
            attributes["api_base"] = self.api_base

        if self.openai_organization:
            attributes["organization"] = self.organization

        if self.openai_proxy:
            attributes["proxy"] = self.proxy

        return attributes


class OpenNVIDIA(NVIDIAMixin, OpenAI):
    
    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.completion

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")

        values = cls._set_client(values)
        return values


class ChatOpenNVIDIA(NVIDIAMixin, ChatOpenAI):
    
    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.chat.completion

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")

        values = cls._set_client(values)
        return values


class OpenNVIDIAEmbeddings(NVIDIAMixin, OpenAIEmbeddings):

    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.embeddings

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        if is_openai_v1():
            default_args: Dict = {"model": self.model, **self.model_kwargs}
        else:
            default_args = {
                "model": self.model,
                "request_timeout": self.request_timeout,
                "headers": self.headers,
                "api_key": self.api_key,
                "organization": self.organization,
                "api_base": self.api_base,
                "api_type": self.api_type,
                "api_version": self.api_version,
                **self.model_kwargs,
            }
        return default_args

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values = cls._set_client(values)
        return values

from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)

class output_parsers:

    __all__ = ["PydanticToolsParser", "JsonOutputToolsParser", "JsonOutputKeyToolsParser"]
