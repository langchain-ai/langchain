import pytest
from langchain_core.embeddings import Embeddings
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_qdrant import QdrantVectorStore


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Mock embed_documents method."""
        return [[1.0, 2.0, 3.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Mock embed_query method."""
        return [1.0, 2.0, 3.0]


@pytest.mark.benchmark
def test_qdrant_vectorstore_init_time(benchmark: BenchmarkFixture) -> None:
    """Test QdrantVectorStore initialization time."""

    def _init_qdrant_vectorstore() -> None:
        for _ in range(10):
            QdrantVectorStore.from_texts(
                texts=["test"],
                embedding=MockEmbeddings(),
                location=":memory:",
                collection_name="test",
            )

    benchmark(_init_qdrant_vectorstore)
