import re
import unittest
from typing import Tuple

import pytest

from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.controller import ToTController
from langchain_experimental.tot.memory import ToTDFSMemory
from langchain_experimental.tot.thought import Thought, ThoughtValidity
from langchain_experimental.tot.thought_generation import SampleCoTStrategy
from tests.unit_tests.fake_llm import FakeLLM

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
    """This is a fake LLM that responds to the sudoku problem."""
    queries = {i: next_step.strip() for i, next_step in enumerate(solutions)}
    return FakeLLM(queries=queries, sequential_responses=True)


class SudokuChecker(ToTChecker):
    def evaluate(
        self, problem_description: str, thoughts: Tuple[str, ...] = ()
    ) -> ThoughtValidity:
        last_thought = thoughts[-1]
        clean_solution = last_thought.replace(" ", "").replace('"', "")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        if sudoku_solution in clean_solution:
            return ThoughtValidity.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID


def test_solve_sudoku(fake_llm_sudoku: FakeLLM) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(
        llm=fake_llm_sudoku,
        checker=SudokuChecker(),
        k=len(solutions),
        c=4,
        tot_strategy_class=SampleCoTStrategy,
    )
    output = tot_chain.run({"problem_description": ""})
    assert output == sudoku_solution


def test_solve_sudoku_k_too_small(fake_llm_sudoku: FakeLLM) -> None:
    """Test simple question that should not need python."""
    tot_chain = ToTChain(
        llm=fake_llm_sudoku,
        checker=SudokuChecker(),
        k=len(solutions) - 1,
        c=4,
        tot_strategy_class=SampleCoTStrategy,
    )
    output = tot_chain.run({"problem_description": ""})
    assert output != sudoku_solution


@pytest.fixture
def fake_llm_checker() -> FakeLLM:
    """This is a fake LLM that responds with a thought validity."""
    responses = [
        "VALID",
        "valid",
        "INVALID",
        "invalid",
        "INTERMEDIATE",
        "intermediate",
        "SOMETHING ELSE",
    ]
    queries = dict(enumerate(responses))
    return FakeLLM(queries=queries, sequential_responses=True)


class ControllerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ToTController(c=3)

    def test_empty(self) -> None:
        memory = ToTDFSMemory([])
        self.assertEqual(self.controller(memory), ())

    def test_one_thoghts(self) -> None:
        thoughts = [Thought(text="a", validity=ThoughtValidity.VALID_FINAL)]
        memory = ToTDFSMemory(thoughts)
        self.assertEqual(self.controller(memory), ("a",))

    def test_two_thoghts(self) -> None:
        memory = ToTDFSMemory(
            [
                Thought(text="a", validity=ThoughtValidity.VALID_INTERMEDIATE),
                Thought(text="b", validity=ThoughtValidity.VALID_INTERMEDIATE),
            ]
        )
        self.assertEqual(self.controller(memory), ("a", "b"))

    def test_two_thoughts_invalid(self) -> None:
        memory = ToTDFSMemory(
            [
                Thought(text="a", validity=ThoughtValidity.VALID_INTERMEDIATE),
                Thought(text="b", validity=ThoughtValidity.INVALID),
            ]
        )
        self.assertEqual(self.controller(memory), ("a",))

    def test_thoughts_rollback(self) -> None:
        a = Thought(text="a", validity=ThoughtValidity.VALID_INTERMEDIATE)
        b = Thought(text="b", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_1 = Thought(text="c_1", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_2 = Thought(text="c_2", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_3 = Thought(text="c_3", validity=ThoughtValidity.VALID_INTERMEDIATE)

        a.children = {b}
        b.children = {c_1, c_2, c_3}

        memory = ToTDFSMemory([a, b, c_3])
        self.assertEqual(self.controller(memory), ("a",))

    def test_thoughts_rollback_invalid(self) -> None:
        a = Thought(text="a", validity=ThoughtValidity.VALID_INTERMEDIATE)
        b = Thought(text="b", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_1 = Thought(text="c_1", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_2 = Thought(text="c_2", validity=ThoughtValidity.VALID_INTERMEDIATE)
        c_3 = Thought(text="c_3", validity=ThoughtValidity.INVALID)

        a.children = {b}
        b.children = {c_1, c_2, c_3}

        memory = ToTDFSMemory([a, b, c_3])
        self.assertEqual(self.controller(memory), ("a",))
