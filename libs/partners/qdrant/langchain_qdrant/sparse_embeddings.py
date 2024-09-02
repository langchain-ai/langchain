from abc import ABC, abstractmethod
from typing import List

from langchain_core.runnables.config import run_in_executor
from pydantic import BaseModel, Field


class SparseVector(BaseModel, extra="forbid"):
    """
    Sparse vector structure
    """

    indices: List[int] = Field(..., description="indices must be unique")
    values: List[float] = Field(
        ..., description="values and indices must be the same length"
    )


class SparseEmbeddings(ABC):
    """An interface for sparse embedding models to use with Qdrant."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> SparseVector:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)
