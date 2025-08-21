from __future__ import annotations

import pytest
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
)

from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests


class TestInMemoryVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return InMemoryVectorStore(embedding=embeddings)


class WithoutGetByIdsVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore that does not implement get_by_ids."""

    get_by_ids = VectorStore.get_by_ids


class TestWithoutGetByIdVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        embeddings = self.get_embeddings()
        return WithoutGetByIdsVectorStore(embedding=embeddings)

    @property
    def has_get_by_ids(self) -> bool:
        return False

    def test_get_by_ids_fails(self, vectorstore: VectorStore) -> None:
        with pytest.raises(
            NotImplementedError,
            match="WithoutGetByIdVectorStore does not yet support get_by_ids.",
        ):
            vectorstore.get_by_ids(["id1", "id2"])
