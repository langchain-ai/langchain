import logging
import os
from typing import Iterable, List, Optional

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
            response = self._client.embeddings.create(
                model=self.model,
                inputs=texts[i : i + self.batch_size],
                parameters={"input_type": "passage", "truncation": "END"},
            )
            embeddings.extend([r["values"] for r in response])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._client.embeddings.create(
            model=self.model,
            inputs=[text],
            parameters={"input_type": "query", "truncation": "END"}
        )[0]["values"]
