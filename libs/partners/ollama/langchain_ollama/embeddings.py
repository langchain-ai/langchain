from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
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

    headers: Optional[dict] = None
    """Additional headers to pass to endpoint (e.g. Authorization, Referer).
    This is useful when Ollama is hosted on cloud services that require
    tokens for authentication.
    """

    auth: Union[Callable, Tuple, None] = None
    """Additional auth tuple or callable to enable Basic/Digest/Custom HTTP Auth.
    Expects the same format, type and values as requests.request auth parameter."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = Client(
            host=self.base_url, headers=self.headers, auth=self.auth
        ).embed(self.model, texts)["embeddings"]
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = (
            await AsyncClient(
                host=self.base_url, headers=self.headers, auth=self.auth
            ).embed(self.model, texts)
        )["embeddings"]
        return embedded_docs

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
