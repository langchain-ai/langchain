import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_exa import ExaSearchRetriever


@pytest.mark.benchmark
def test_exa_retriever_init_time(benchmark: BenchmarkFixture) -> None:
    """Test ExaSearchRetriever initialization time."""

    def _init_exa_retriever() -> None:
        for _ in range(10):
            ExaSearchRetriever(api_key="test-key")

    benchmark(_init_exa_retriever)
