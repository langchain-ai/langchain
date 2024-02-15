import hashlib
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class FakeEmbeddings(Embeddings, BaseModel):
    """Fake embedding model."""

    size: int
    """The size of the embedding vector."""

    def _get_embedding(self) -> List[float]:
        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding()


class DeterministicFakeEmbedding(Embeddings, BaseModel):
    """
    Fake embedding model that always returns
    the same embedding vector for the same text.
    """

    size: int
    """The size of the embedding vector."""

    def _get_embedding(self, seed: int) -> List[float]:
        # set the seed for the random generator
        np.random.seed(seed)
        return list(np.random.normal(size=self.size))

    def _get_seed(self, text: str) -> int:
        """
        Get a seed for the random generator, using the hash of the text.
        """
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(seed=self._get_seed(_)) for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(seed=self._get_seed(text))
