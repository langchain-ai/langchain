from typing import (
    List,
    Optional,
)

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from ollama import AsyncClient, Client


class OllamaEmbeddings(BaseModel, Embeddings):
    """OllamaEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaEmbeddings

            embedder = OllamaEmbeddings(model="llama3")
            embedder.embed_query("what is the place that jonathan worked at?")
    """

    model: str
    """Model name to use."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx Client. 
    For a full list of the params, see [this link](https://pydoc.dev/httpx/latest/httpx.Client.html)
    """

    _client: Client = Field(default=None)
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = Field(default=None)
    """
    The async client to use for making requests.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=False, skip_on_failure=True)
    def _set_clients(cls, values: dict) -> dict:
        """Set clients to use for ollama."""
        values["_client"] = Client(host=values["base_url"], **values["client_kwargs"])
        values["_async_client"] = AsyncClient(
            host=values["base_url"], **values["client_kwargs"]
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = self._client.embed(self.model, texts)["embeddings"]
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = (await self._async_client.embed(self.model, texts))[
            "embeddings"
        ]
        return embedded_docs

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
