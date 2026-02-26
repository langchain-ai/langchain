import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import]

from langchain_prompty import create_chat_prompt


@pytest.mark.benchmark
def test_create_chat_prompt_init_time(benchmark: BenchmarkFixture) -> None:
    """Test create_chat_prompt initialization time."""

    def _create_chat_prompts() -> None:
        for _ in range(10):
            create_chat_prompt("Hello world")

    benchmark(_create_chat_prompts)
