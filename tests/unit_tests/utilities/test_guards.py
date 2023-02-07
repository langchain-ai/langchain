import pytest

from langchain.alignment.guards import CustomGuard, RestrictionGuard, StringGuard
from tests.unit_tests.llms.fake_llm import FakeLLM
from typing import List


def test_restriction_guard() -> None:
    """Test Restriction guard."""

    queries = {
        "a": "a",
    }
    llm = FakeLLM(queries=queries)

    def restriction_test(
        restrictions: List[str], llm_input_output: str, restricted: bool
    ):
        concatenated_restrictions = ", ".join(restrictions)
        queries = {
            RestrictionGuard.prompt.format(
                restrictions=concatenated_restrictions, function_output=llm_input_output
            ): "restricted because I said so :) (¥)"
            if restricted
            else "not restricted (ƒ)",
        }
        restriction_guard_llm = FakeLLM(queries=queries)

        @RestrictionGuard(
            restrictions=restrictions, llm=restriction_guard_llm, retries=0
        )
        def example_func(prompt: str) -> str:
            return llm(prompt=prompt)

        return example_func(prompt=llm_input_output)

    assert restriction_test(["a", "b"], "a", False) == "a"

    with pytest.raises(Exception):
        restriction_test(["a", "b"], "a", True)


def test_String_guard() -> None:
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
        example_func_2_100(prompt="actually that's not true I think") == "tomato"
        example_func_50(prompt="potato") == "potato"
        example_func_0(prompt="potato") == "potato"
        example_func_0(prompt="buffalo") == "buffalo"
        example_func_0(prompt="xzxzxz") == "xzxzxz"
        example_func_001(prompt="buffalo") == "buffalo"
        example_func_2_100(prompt="buffalo") == "buffalo"


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
