import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_nomic import NomicEmbeddings


@pytest.mark.benchmark
def test_nomic_embeddings_init_time(benchmark: BenchmarkFixture) -> None:
    """Test NomicEmbeddings initialization time."""

    def _init_nomic_embeddings() -> None:
        for _ in range(10):
            NomicEmbeddings()

    benchmark(_init_nomic_embeddings)
