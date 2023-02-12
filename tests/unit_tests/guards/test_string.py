import pytest

from langchain.guards.string import StringGuard
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_string_guard() -> None:
    """Test String guard."""

    queries = {
        "tomato": "tomato",
        "potato": "potato",
        "buffalo": "buffalo",
        "xzxzxz": "xzxzxz",
        "buffalos eat lots of potatos": "potato",
        "actually that's not true I think": "tomato",
    }

    llm = FakeLLM(queries=queries)

    @StringGuard(protected_strings=["tomato"], leniency=1, retries=0)
    def example_func_100(prompt: str) -> str:
        return llm(prompt=prompt)

    @StringGuard(protected_strings=["tomato", "buffalo"], leniency=1, retries=0)
    def example_func_2_100(prompt: str) -> str:
        return llm(prompt=prompt)

    @StringGuard(protected_strings=["tomato"], leniency=0.5, retries=0)
    def example_func_50(prompt: str) -> str:
        return llm(prompt)

    @StringGuard(protected_strings=["tomato"], leniency=0, retries=0)
    def example_func_0(prompt: str) -> str:
        return llm(prompt)

    @StringGuard(protected_strings=["tomato"], leniency=0.01, retries=0)
    def example_func_001(prompt: str) -> str:
        return llm(prompt)

    assert example_func_100(prompt="potato") == "potato"
    assert example_func_50(prompt="buffalo") == "buffalo"
    assert example_func_001(prompt="xzxzxz") == "xzxzxz"
    assert example_func_2_100(prompt="xzxzxz") == "xzxzxz"
    assert example_func_100(prompt="buffalos eat lots of potatos") == "potato"

    with pytest.raises(Exception):
        example_func_2_100(prompt="actually that's not true I think")
    assert example_func_50(prompt="potato") == "potato"
    with pytest.raises(Exception):
        example_func_0(prompt="potato")
    with pytest.raises(Exception):
        example_func_0(prompt="buffalo")
    with pytest.raises(Exception):
        example_func_0(prompt="xzxzxz")
    assert example_func_001(prompt="buffalo") == "buffalo"
    with pytest.raises(Exception):
        example_func_2_100(prompt="buffalo")
