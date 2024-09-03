"""Module contains a few fake embedding models for testing purposes."""

# Please do not add additional fake embedding model implementations here.
import hashlib
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


class FakeEmbeddings(Embeddings, BaseModel):
    """Fake embedding model for unit testing purposes.

    This embedding model creates embeddings by sampling from a normal distribution.

    Do not use this outside of testing, as it is not a real embedding model.

    Instantiate:
        .. code-block:: python

            from langchain_core.embeddings import FakeEmbeddings
            embed = FakeEmbeddings(size=100)

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.700234640213188, -0.581266257710429, -1.1328482266445354]

    Embed multiple texts:
        .. code-block:: python

            input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
    """

    size: int
    """The size of the embedding vector."""

    def _get_embedding(self) -> List[float]:
        import numpy as np  # type: ignore[import-not-found, import-untyped]

        return list(np.random.normal(size=self.size))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding()


class DeterministicFakeEmbedding(Embeddings, BaseModel):
    """Deterministic fake embedding model for unit testing purposes.

    This embedding model creates embeddings by sampling from a normal distribution
    with a seed based on the hash of the text.

    Do not use this outside of testing, as it is not a real embedding model.

    Instantiate:
        .. code-block:: python

            from langchain_core.embeddings import DeterministicFakeEmbedding
            embed = DeterministicFakeEmbedding(size=100)

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.700234640213188, -0.581266257710429, -1.1328482266445354]

    Embed multiple texts:
        .. code-block:: python

            input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
    """

    size: int
    """The size of the embedding vector."""

    def _get_embedding(self, seed: int) -> List[float]:
        import numpy as np  # type: ignore[import-not-found, import-untyped]

        # set the seed for the random generator
        np.random.seed(seed)
        return list(np.random.normal(size=self.size))

    def _get_seed(self, text: str) -> int:
        """Get a seed for the random generator, using the hash of the text."""
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(seed=self._get_seed(_)) for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(seed=self._get_seed(text))
