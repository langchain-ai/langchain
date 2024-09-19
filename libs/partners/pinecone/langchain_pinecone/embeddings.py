import logging
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


class PineconeEmbeddings(BaseModel, Embeddings):
    """PineconeEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_pinecone import PineconeEmbeddings

            model = PineconeEmbeddings(model="multilingual-e5-large")
    """

    # Clients
    _client: PineconeClient = PrivateAttr(default=None)
    _async_client: aiohttp.ClientSession = PrivateAttr(default=None)
    model: str
    """Model to use for example 'multilingual-e5-large'."""
    # Config
    batch_size: Optional[int] = None
    """Batch size for embedding documents."""
    query_params: Dict = Field(default_factory=dict)
    """Parameters for embedding query."""
    document_params: Dict = Field(default_factory=dict)
    """Parameters for embedding document"""
    #
    dimension: Optional[int] = None
    #
    show_progress_bar: bool = False
    pinecone_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "PINECONE_API_KEY",
            error_message="Pinecone API key not found. Please set the PINECONE_API_KEY "
            "environment variable or pass it via `pinecone_api_key`.",
        ),
        alias="api_key",
    )
    """Pinecone API key. 
    
    If not provided, will look for the PINECONE_API_KEY environment variable."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_config(cls, values: dict) -> Any:
        """Set default configuration based on model."""
        default_config_map = {
            "multilingual-e5-large": {
                "batch_size": 96,
                "query_params": {"input_type": "query", "truncation": "END"},
                "document_params": {"input_type": "passage", "truncation": "END"},
                "dimension": 1024,
            }
        }
        model = values.get("model")
        if model in default_config_map:
            config = default_config_map[model]
            for key, value in config.items():
                if key not in values:
                    values[key] = value
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that Pinecone version and credentials exist in environment."""
        api_key_str = self.pinecone_api_key.get_secret_value()
        client = PineconeClient(api_key=api_key_str, source_tag="langchain")
        self._client = client

        # initialize async client
        if not self._async_client:
            self._async_client = aiohttp.ClientSession(
                headers={
                    "Api-Key": api_key_str,
                    "Content-Type": "application/json",
                    "X-Pinecone-API-Version": "2024-07",
                }
            )
        return self

    def _get_batch_iterator(self, texts: List[str]) -> Iterable:
        if self.batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        else:
            batch_size = self.batch_size

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), batch_size))
        else:
            _iter = range(0, len(texts), batch_size)

        return _iter

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings: List[List[float]] = []

        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            response = self._client.inference.embed(
                model=self.model,
                parameters=self.document_params,
                inputs=texts[i : i + self.batch_size],
            )
            embeddings.extend([r["values"] for r in response])

        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            response = await self._aembed_texts(
                model=self.model,
                parameters=self.document_params,
                texts=texts[i : i + self.batch_size],
            )
            embeddings.extend([r["values"] for r in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._client.inference.embed(
            model=self.model, parameters=self.query_params, inputs=[text]
        )[0]["values"]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed query text."""
        response = await self._aembed_texts(
            model=self.model,
            parameters=self.document_params,
            texts=[text],
        )
        return response["data"][0]["values"]

    async def _aembed_texts(
        self, texts: List[str], model: str, parameters: dict
    ) -> Dict:
        data = {
            "model": model,
            "inputs": [{"text": text} for text in texts],
            "parameters": parameters,
        }
        async with self._async_client.post(
            "https://api.pinecone.io/embed", json=data
        ) as response:
            response_data = await response.json(content_type=None)
            return response_data
