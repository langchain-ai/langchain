"""Simple benchmark for SERPEX search."""

import pytest
from pydantic import SecretStr

from langchain_serpex import SerpexSearchResults


@pytest.mark.benchmark
def test_serpex_initialization_benchmark(benchmark: pytest.fixture) -> None:
    """Benchmark tool initialization."""

    def create_tool() -> SerpexSearchResults:
        return SerpexSearchResults(api_key=SecretStr("test_key"))

    result = benchmark(create_tool)
    assert result.name == "serpex_search"


@pytest.mark.benchmark
def test_serpex_params_building_benchmark(benchmark: pytest.fixture) -> None:
    """Benchmark parameter building."""
    tool = SerpexSearchResults(api_key=SecretStr("test_key"))

    def build_params() -> dict:
        return tool._build_params("test query", engine="google", time_range="day")

    result = benchmark(build_params)
    assert result["q"] == "test query"
