"""Standard unit tests for ExaSearchRetriever."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_exa import ExaFindSimilarResults, ExaSearchResults, ExaSearchRetriever
from langchain_exa._utilities import EXA_INTEGRATION_HEADER, EXA_INTEGRATION_NAME


@pytest.mark.benchmark
def test_exa_retriever_init_time(benchmark: BenchmarkFixture) -> None:
    """Test ExaSearchRetriever initialization time."""

    def _init_exa_retriever() -> None:
        for _ in range(10):
            ExaSearchRetriever()

    benchmark(_init_exa_retriever)


def test_exa_clients_include_integration_header() -> None:
    """Test Exa clients include the LangChain integration header."""
    clients = [
        ExaSearchRetriever().client,
        ExaSearchResults().client,
        ExaFindSimilarResults().client,
    ]

    for client in clients:
        assert client.headers[EXA_INTEGRATION_HEADER] == EXA_INTEGRATION_NAME


def test_exa_retriever_search_type_values() -> None:
    """Test ExaSearchRetriever supports current search type values."""
    for search_type in ("auto", "deep", "fast"):
        retriever = ExaSearchRetriever(type=search_type)
        assert retriever.type == search_type
