from typing import List
from unittest.mock import Mock

from langchain_core.embeddings import Embeddings

from langchain_astradb.vectorstores import AstraDBVectorStore


class SomeEmbeddings(Embeddings):
    """
    Turn a sentence into an embedding vector in some way.
    Not important how. It is deterministic is all that counts.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        unnormed0 = [ord(c) for c in text[: self.dimension]]
        unnormed = (unnormed0 + [1] + [0] * (self.dimension - 1 - len(unnormed0)))[
            : self.dimension
        ]
        norm = sum(x * x for x in unnormed) ** 0.5
        normed = [x / norm for x in unnormed]
        return normed

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    mock_astra_db = Mock()
    embedding = SomeEmbeddings(dimension=2)
    AstraDBVectorStore(
        embedding=embedding,
        collection_name="mock_coll_name",
        astra_db_client=mock_astra_db,
    )
