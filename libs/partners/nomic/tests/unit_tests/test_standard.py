"""Unit tests for standard tests in Nomic partner integration."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import]

from langchain_nomic import NomicEmbeddings


@pytest.mark.benchmark
def test_nomic_embeddings_init_time(benchmark: BenchmarkFixture) -> None:
    """Test NomicEmbeddings initialization time."""

    def _init_nomic_embeddings() -> None:
        for _ in range(10):
            NomicEmbeddings(model="test")

    benchmark(_init_nomic_embeddings)
