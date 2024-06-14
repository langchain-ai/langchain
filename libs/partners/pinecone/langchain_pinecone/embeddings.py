import logging
import os
from typing import Dict, Iterable, List, Optional

import aiohttp
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import convert_to_secret_str
from pinecone import Pinecone as PineconeClient  # type: ignore

logger = logging.getLogger(__name__)


class PineconeEmbeddings(BaseModel, Embeddings):
    """PineconeEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_pinecone import PineconeEmbeddings

            model = PineconeEmbeddings()
    """

    _client: PineconeClient = Field(exclude=True)
    _async_client: aiohttp.ClientSession = Field(exclude=True)
    model: str
    batch_size: int
    show_progress_bar: bool = False
    truncation: Optional[bool] = None
    pinecone_api_key: Optional[SecretStr] = None

    class Config:
        extra = Extra.forbid

    @root_validator(pre=True)
    def default_values(cls, values: dict) -> dict:
        """Set default batch size"""

        batch_size = values.get("batch_size")
        if batch_size is None:
            values["batch_size"] = 128
        return values

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        """Validate that Pinecone credentials exist in environment."""
        pinecone_api_key = values.get("pinecone_api_key") or os.getenv(
            "PINECONE_API_KEY", None
        )
        if pinecone_api_key:
            api_key_secretstr = convert_to_secret_str(pinecone_api_key)
            values["pinecone_api_key"] = api_key_secretstr

            api_key_str = api_key_secretstr.get_secret_value()
        else:
            api_key_str = None
        values["_client"] = PineconeClient(api_key=api_key_str, source_tag="langchain")

        # initialize async client
        if not values.get("_async_client"):
            if api_key_str is None:
                raise ValueError(
                    "Pinecone API key not found. Please set the PINECONE_API_KEY "
                    "environment variable or pass it via `pinecone_api_key`."
                )
            values["_async_client"] = aiohttp.ClientSession(
                headers={
                    "Api-Key": api_key_str,
                    "Content-Type": "application/json",
                    "X-Pinecone-API-Version": "2024-07",
                }
            )
        return values

    def _get_batch_iterator(self, texts: List[str]) -> Iterable:
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), self.batch_size))
        else:
            _iter = range(0, len(texts), self.batch_size)  # type: ignore

        return _iter

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings: List[List[float]] = []

        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            response = self._client.inference.embed(
                model=self.model,
                inputs=texts[i : i + self.batch_size],
                parameters={"input_type": "passage", "truncation": "END"},
            )
            embeddings.extend([r["values"] for r in response])

        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            response = await self._aembed_texts(
                texts=texts[i : i + self.batch_size],
                model=self.model,
                parameters={"input_type": "passage", "truncation": "END"},
            )
            embeddings.extend([r["values"] for r in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._client.inference.embed(
            model=self.model,
            inputs=[text],
            parameters={"input_type": "query", "truncation": "END"},
        )[0]["values"]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed query text."""
        response = await self._aembed_texts(
            texts=[text],
            model=self.model,
            parameters={"input_type": "query", "truncation": "END"},
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
