import json
from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import BaseModel, Field, SecretStr, model_validator


class HunyuanEmbeddings(Embeddings, BaseModel):
    """Tencent Hunyuan embedding models API by Tencent.

    For more information, see https://cloud.tencent.com/document/product/1729
    """

    hunyuan_secret_id: Optional[SecretStr] = Field(alias="secret_id", default=None)
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[SecretStr] = Field(alias="secret_key", default=None)
    """Hunyuan Secret Key"""
    region: Literal["ap-guangzhou", "ap-beijing"] = "ap-guangzhou"
    """The region of hunyuan service."""
    embedding_ctx_length: int = 1024
    """The max embedding context length of hunyuan embedding (defaults to 1024)."""
    show_progress_bar: bool = False
    """Show progress bar when embedding. Default is False."""

    client: Any = Field(default=None, exclude=True)
    """The tencentcloud client."""
    request_cls: Optional[Type] = Field(default=None, exclude=True)
    """The request class of tencentcloud sdk."""

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["hunyuan_secret_id"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "hunyuan_secret_id",
                "HUNYUAN_SECRET_ID",
            )
        )
        values["hunyuan_secret_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "hunyuan_secret_key",
                "HUNYUAN_SECRET_KEY",
            )
        )

        try:
            from tencentcloud.common.credential import Credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.hunyuan.v20230901.hunyuan_client import HunyuanClient
            from tencentcloud.hunyuan.v20230901.models import GetEmbeddingRequest
        except ImportError:
            raise ImportError(
                "Could not import tencentcloud sdk python package. Please install it "
                'with `pip install "tencentcloud-sdk-python>=3.0.1139"`.'
            )

        client_profile = ClientProfile()
        client_profile.httpProfile.pre_conn_pool_size = 3

        credential = Credential(
            values["hunyuan_secret_id"].get_secret_value(),
            values["hunyuan_secret_key"].get_secret_value(),
        )

        values["request_cls"] = GetEmbeddingRequest

        values["client"] = HunyuanClient(credential, values["region"], client_profile)
        return values

    def _embed_text(self, text: str) -> List[float]:
        if self.request_cls is None:
            raise AssertionError("Request class is not initialized.")
        request = self.request_cls()
        request.Input = text

        response = self.client.GetEmbedding(request)

        _response: Dict[str, Any] = json.loads(response.to_json_string())

        data: Optional[List[Dict[str, Any]]] = _response.get("Data")
        if not data:
            raise RuntimeError("Occur hunyuan embedding error: Data is empty")

        embedding = data[0].get("Embedding")
        if not embedding:
            raise RuntimeError("Occur hunyuan embedding error: Embedding is empty")

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
            except ImportError as e:
                raise ImportError(
                    "Package tqdm must be installed if show_progress_bar=True. "
                    "Please install with 'pip install tqdm' or set "
                    "show_progress_bar=False."
                ) from e
            _iter = tqdm(iterable=texts, desc="Hunyuan Embedding")
        else:
            _iter = texts
        for text in _iter:
            embeddings.append(self.embed_query(text))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)
