from typing import List

import ollama
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra


class OllamaEmbeddings(BaseModel, Embeddings):
    """OllamaEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaEmbeddings

            model = OllamaEmbeddings()
    """

    model: str = "llama2"
    """Model name to use."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embedded_docs = []
        for doc in texts:
            embedded_docs.append(list(ollama.embeddings(self.model, doc)["embedding"]))
        return embedded_docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    # only keep aembed_documents and aembed_query if they're implemented!
    # delete them otherwise to use the base class' default
    # implementation, which calls the sync version in an executor
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
