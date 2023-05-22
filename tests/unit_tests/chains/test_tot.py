import pytest

from langchain.chains.tot.base import ToTChain, ToTChecker, SolutionType
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def fake_llm_counter() -> FakeLLM:
    """This is a fake LLM that counts to 100."""
    queries = {i: f'```\n{{"next_step": "{i}"}}\n```' for i in range(101)}
    return FakeLLM(queries=queries, sequential_responses=True)


class TestChecker25(ToTChecker):
    def evaluate(self, problem_description: str, solution: str) -> SolutionType:
        if "25" in solution:
            return SolutionType.VALID_FINAL
        elif "2" in solution or "5" in solution:
            return SolutionType.VALID_INTERMEDIATE
        return SolutionType.INVALID


def test_guess_25(fake_llm_counter: ToTChain) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(llm=fake_llm_counter, checker=TestChecker25(), k=26)
    question = "Guess My Number!"
    output = tot_chain.run({"problem_description": question})
    assert output == "25"


def test_guess_k_too_small(fake_llm_counter: ToTChain) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(llm=fake_llm_counter, checker=TestChecker25(), k=24)
    question = "Guess My Number!"
    output = tot_chain.run({"problem_description": question})
    assert output != "25"
