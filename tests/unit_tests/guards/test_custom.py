import pytest

from langchain.guards.custom import CustomGuard
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_custom_guard() -> None:
    """Test custom guard."""

    queries = {
        "tomato": "tomato",
        "potato": "potato",
    }

    llm = FakeLLM(queries=queries)

    def starts_with_t(prompt: str) -> bool:
        return prompt.startswith("t")

    @CustomGuard(guard_function=starts_with_t, retries=0)
    def example_func(prompt: str) -> str:
        return llm(prompt=prompt)

    assert example_func(prompt="potato") == "potato"

    with pytest.raises(Exception):
        assert example_func(prompt="tomato") == "tomato"
