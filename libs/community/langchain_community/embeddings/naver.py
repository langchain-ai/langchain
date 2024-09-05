import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    SecretStr,
)
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

DEFAULT_BASE_URL = "https://clovastudio.apigw.ntruss.com"

logger = logging.getLogger(__name__)


def _raise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = response.read().decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


async def _araise_on_error(response: httpx.Response) -> None:
    """Raise an error if the response is an error."""
    if httpx.codes.is_error(response.status_code):
        error_message = (await response.aread()).decode("utf-8")
        raise httpx.HTTPStatusError(
            f"Error response {response.status_code} "
            f"while fetching {response.url}: {error_message}",
            request=response.request,
            response=response,
        )


class ClovaXEmbeddings(BaseModel, Embeddings):
    """`NCP ClovaStudio` Embedding API.

    following environment variables set or passed in constructor in lower case:
    - ``NCP_CLOVASTUDIO_API_KEY``
    - ``NCP_APIGW_API_KEY``
    - ``NCP_CLOVASTUDIO_APP_ID``

    Example:
        .. code-block:: python

            from langchain_community import ClovaXEmbeddings

            model = ClovaXEmbeddings(model="clir-emb-dolphin")
            output = embedding.embed_documents(documents)
    """  # noqa: E501

    client: httpx.Client = Field(default=None)  #: :meta private:
    async_client: httpx.AsyncClient = Field(default=None)  #: :meta private:

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    base_url: Optional[str] = Field(default=None, alias="base_url")
    """
    Automatically inferred from env are  `NCP_CLOVASTUDIO_API_BASE_URL` if not provided.
    """

    app_id: Optional[str] = Field(default=None, alias="clovastudio_app_id")
    service_app: bool = Field(
        default=False,
        description="false: use testapp, true: use service app on NCP Clova Studio",
    )
    model_name: str = Field(
        default="clir-emb-dolphin",
        alias="model",
        description="NCP ClovaStudio embedding model name",
    )

    timeout: int = 60

    class Config:
        arbitrary_types_allowed = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY",
            "ncp_apigw_api_key": "NCP_APIGW_API_KEY",
        }

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the client."""
        return self._default_params

    @property
    def _api_url(self) -> str:
        """GET embedding api url"""
        app_type = "serviceapp" if self.service_app else "testapp"
        model_name = self.model_name if self.model_name != "bge-m3" else "v2"
        return (
            f"{self.base_url}/{app_type}"
            f"/v1/api-tools/embedding/{model_name}/{self.app_id}"
        )

    def __init__(self, **kwargs: Any) -> None:
        """Validate that api key and python package exists in environment."""
        kwargs["ncp_clovastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "api_key", "NCP_CLOVASTUDIO_API_KEY")
        )
        kwargs["ncp_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                kwargs, "apigw_api_key", "NCP_APIGW_API_KEY", "ncp_apigw_api_key"
            )
        )
        kwargs["base_url"] = get_from_dict_or_env(
            kwargs, "base_url", "NCP_CLOVASTUDIO_API_BASE_URL", DEFAULT_BASE_URL
        )

        super().__init__(**kwargs)

        self.timeout = 90 if self.timeout is None else self.timeout

        self.app_id = get_from_dict_or_env(kwargs, "app_id", "NCP_CLOVASTUDIO_APP_ID")

        if "client" in kwargs:
            self.client = kwargs.get("client")
        else:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers=self.default_headers(kwargs),
                timeout=self.timeout,
            )

        if "async_client" in kwargs:
            self.async_client = kwargs.get("async_client")
        else:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers(kwargs),
                timeout=self.timeout,
            )

    @staticmethod
    def default_headers(values):
        clovastudio_api_key = (
            values["ncp_clovastudio_api_key"].get_secret_value()
            if values["ncp_clovastudio_api_key"]
            else None
        )
        apigw_api_key = (
            values["ncp_apigw_api_key"].get_secret_value()
            if values["ncp_apigw_api_key"]
            else None
        )
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-NCP-CLOVASTUDIO-API-KEY": clovastudio_api_key,
            "X-NCP-APIGW-API-KEY": apigw_api_key,
        }

    def _embed_text(self, text: str) -> List[float]:
        payload = {"text": text}
        response = self.client.post(url=self._api_url, json=payload)
        _raise_on_error(response)
        return response.json()["result"]["embedding"]

    async def _aembed_text(self, text: str) -> List[float]:
        payload = {"text": text}
        response = await self.async_client.post(url=self._api_url, json=payload)
        await _araise_on_error(response)
        return response.json()["result"]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = await self._aembed_text(text)
            embeddings.append(embedding)
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        return await self._aembed_text(text)
