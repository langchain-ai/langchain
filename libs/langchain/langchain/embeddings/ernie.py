from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ErnieEmbeddings(BaseModel, Embeddings):
    """ERNIE embedding models.

    To use, you should have the ``erniebot`` python package installed, and the
    environment variable ``EB_ACCESS_TOKEN`` set with your AI Studio access
    token.

    Example:
        .. code-block:: python
            from langchain.embeddings import ErnieEmbeddings
            ernie_embeddings = ErnieEmbeddings()
    """

    client: Any = None
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    chunk_size: int = 16
    """Chunk size to use when the input is a list of texts."""
    aistudio_access_token: Optional[str] = None
    """AI Studio access token."""
    model: str = "ernie-text-embedding"
    """Model to use."""
    request_timeout: Optional[int] = 60
    """How many seconds to wait for the server to send data before giving up."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["aistudio_access_token"] = get_from_dict_or_env(
            values,
            "aistudio_access_token",
            "EB_ACCESS_TOKEN",
        )
        try:
            import erniebot

            values["client"] = erniebot.Embedding
        except ImportError:
            raise ImportError(
                "Could not import erniebot python package. "
                "Please install it with `pip install erniebot`."
            )
        return values

    def embed_query(self, text: str) -> List[float]:
        resp = self.embed_documents([text])
        return resp[0]

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        lst = []
        for chunk in text_in_chunks:
            resp = self.client.create(
                _config_=self._get_auth_config(), input=chunk, model=self.model
            )
            lst.extend([res["embedding"] for res in resp["data"]])
        return lst

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        lst = []
        for chunk in text_in_chunks:
            resp = await self.client.acreate(
                _config_=self._get_auth_config(), input=chunk, model=self.model
            )
            for res in resp["data"]:
                lst.extend([res["embedding"]])
        return lst

    def _get_auth_config(self) -> dict:
        return {"api_type": "aistudio", "access_token": self.aistudio_access_token}
