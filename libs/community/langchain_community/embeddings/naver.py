import logging
from typing import Any, Dict, List, Optional, cast

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_env
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

_DEFAULT_BASE_URL = "https://clovastudio.apigw.ntruss.com"

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

    client: Optional[httpx.Client] = Field(default=None)  #: :meta private:
    async_client: Optional[httpx.AsyncClient] = Field(default=None)  #: :meta private:

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    base_url: Optional[str] = Field(default=None, alias="base_url")
    """
    Automatically inferred from env are  `NCP_CLOVASTUDIO_API_BASE_URL` if not provided.
    """

    app_id: Optional[str] = Field(default=None)
    service_app: bool = Field(
        default=False,
        description="false: use testapp, true: use service app on NCP Clova Studio",
    )
    model_name: str = Field(
        default="clir-emb-dolphin",
        validation_alias=AliasChoices("model_name", "model"),
        description="NCP ClovaStudio embedding model name",
    )

    timeout: int = Field(gt=0, default=60)

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY",
            "ncp_apigw_api_key": "NCP_APIGW_API_KEY",
        }

    @property
    def _api_url(self) -> str:
        """GET embedding api url"""
        app_type = "serviceapp" if self.service_app else "testapp"
        model_name = self.model_name if self.model_name != "bge-m3" else "v2"
        return (
            f"{self.base_url}/{app_type}"
            f"/v1/api-tools/embedding/{model_name}/{self.app_id}"
        )

    @model_validator(mode="after")
    def validate_model_after(self) -> Self:
        if not self.ncp_clovastudio_api_key:
            self.ncp_clovastudio_api_key = convert_to_secret_str(
                get_from_env("ncp_clovastudio_api_key", "NCP_CLOVASTUDIO_API_KEY")
            )

        if not self.ncp_apigw_api_key:
            self.ncp_apigw_api_key = convert_to_secret_str(
                get_from_env("ncp_apigw_api_key", "NCP_APIGW_API_KEY", "")
            )

        if not self.base_url:
            self.base_url = get_from_env(
                "base_url", "NCP_CLOVASTUDIO_API_BASE_URL", _DEFAULT_BASE_URL
            )

        if not self.app_id:
            self.app_id = get_from_env("app_id", "NCP_CLOVASTUDIO_APP_ID")

        if not self.client:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers=self.default_headers(),
                timeout=self.timeout,
            )

        if not self.async_client:
            self.async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers(),
                timeout=self.timeout,
            )

        return self

    def default_headers(self) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        clovastudio_api_key = (
            self.ncp_clovastudio_api_key.get_secret_value()
            if self.ncp_clovastudio_api_key
            else None
        )
        if clovastudio_api_key:
            headers["X-NCP-CLOVASTUDIO-API-KEY"] = clovastudio_api_key

        apigw_api_key = (
            self.ncp_apigw_api_key.get_secret_value()
            if self.ncp_apigw_api_key
            else None
        )
        if apigw_api_key:
            headers["X-NCP-APIGW-API-KEY"] = apigw_api_key

        return headers

    def _embed_text(self, text: str) -> List[float]:
        payload = {"text": text}
        client = cast(httpx.Client, self.client)
        response = client.post(url=self._api_url, json=payload)
        _raise_on_error(response)
        return response.json()["result"]["embedding"]

    async def _aembed_text(self, text: str) -> List[float]:
        payload = {"text": text}
        async_client = cast(httpx.AsyncClient, self.client)
        response = await async_client.post(url=self._api_url, json=payload)
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
