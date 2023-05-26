import json
import re

import pytest

from langchain.chains.tot.base import SolutionType, ToTChain, ToTChecker
from tests.unit_tests.llms.fake_llm import FakeLLM

sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
solutions = [
    "3,*,4,2|1,*,3,*|*,1,*,3|4,*,*,1",  # VALID_INTERMEDIATE
    "   3,4,1,2|1,6,3,*|*,1,*,3|4,*,*,1",  # INVALID c=1
    "   3,4,1,2|1,7,3,*|*,1,*,3|4,*,*,1",  # INVALID c=2
    "   3,4,1,2|1,8,3,*|*,1,*,3|4,*,*,1",  # INVALID c=3
    "   3,4,1,2|1,2,3,*|*,1,*,3|4,*,*,1",  # VALID_INTERMEDIATE c=4 (rollback)
    "3,1,4,2|1,*,3,*|*,1,*,3|4,*,*,1",  # INVALID (rollback)
    "3,4,1,2|1,2,3,4|*,1,*,3|4,*,*,1",  # VALID_INTERMEDIATE
    "   3,4,1,2|1,2,3,4|4,1,*,3|4,*,*,1",  # INVALID (rollback)
    "   3,4,1,2|1,2,3,4|2,1,4,3|4,*,*,1",  # VALID_INTERMEDIATE
    "       3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",  # VALID_INTERMEDIATE
    "           3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",  # VALID_FINAL
]
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"


@pytest.fixture
def fake_llm_sudoku() -> FakeLLM:
    """This is a fake LLM that counts to 100."""
    queries = {
        i: json.dumps({"next_step": next_step.strip()})
        for i, next_step in enumerate(solutions)
    }
    return FakeLLM(queries=queries, sequential_responses=True)


class SudokuChecker(ToTChecker):
    def evaluate(self, problem_description: str, solution: str) -> SolutionType:
        clean_solution = solution.replace(" ", "")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        if sudoku_solution in clean_solution:
            return SolutionType.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return SolutionType.VALID_INTERMEDIATE
        else:
            return SolutionType.INVALID


def test_solve_sudoku(fake_llm_sudoku: ToTChain) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(llm=fake_llm_sudoku, checker=SudokuChecker(), k=len(solutions))
    output = tot_chain.run({"problem_description": ""})
    assert output == sudoku_solution


def test_solve_sudoku_k_too_small(fake_llm_sudoku: ToTChain) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(
        llm=fake_llm_sudoku, checker=SudokuChecker(), k=len(solutions) - 1
    )
    output = tot_chain.run({"problem_description": ""})
    assert output != sudoku_solution
