from typing import List
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from langchain_astradb.vectorstores import (
    DEFAULT_INDEXING_OPTIONS,
    AstraDBVectorStore,
)


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


class TestAstraDB:
    def test_initialization(self) -> None:
        """Test integration vectorstore initialization."""
        mock_astra_db = Mock()
        embedding = SomeEmbeddings(dimension=2)
        AstraDBVectorStore(
            embedding=embedding,
            collection_name="mock_coll_name",
            astra_db_client=mock_astra_db,
        )

    def test_astradb_vectorstore_unit_indexing_normalization(self) -> None:
        """Unit test of the indexing policy normalization"""
        n3_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
        )
        assert n3_idx == DEFAULT_INDEXING_OPTIONS

        al_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=["a1", "a2"],
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
        )
        assert al_idx == {"allow": ["metadata.a1", "metadata.a2"]}

        dl_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=["d1", "d2"],
            collection_indexing_policy=None,
        )
        assert dl_idx == {"deny": ["metadata.d1", "metadata.d2"]}

        custom_policy = {
            "deny": ["myfield", "other_field.subfield", "metadata.long_text"]
        }
        cip_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=custom_policy,
        )
        assert cip_idx == custom_policy

        with pytest.raises(ValueError):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=["b"],
                collection_indexing_policy=None,
            )

        with pytest.raises(ValueError):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=None,
                collection_indexing_policy={"a": "z"},
            )

        with pytest.raises(ValueError):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=None,
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
            )

        with pytest.raises(ValueError):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
            )
