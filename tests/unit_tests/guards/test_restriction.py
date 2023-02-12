from typing import List

import pytest

from langchain.guards.restriction import RestrictionGuard
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_restriction_guard() -> None:
    """Test Restriction guard."""

    queries = {
        "a": "a",
    }
    llm = FakeLLM(queries=queries)

    def restriction_test(
        restrictions: List[str], llm_input_output: str, restricted: bool
    ) -> str:
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
