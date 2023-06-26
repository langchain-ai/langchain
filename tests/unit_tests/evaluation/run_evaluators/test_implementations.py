"""Test run evaluator implementations basic functionality."""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from langchain.evaluation.run_evaluators import get_criteria_evaluator, get_qa_evaluator
from tests.unit_tests.llms.fake_llm import FakeLLM

if TYPE_CHECKING:
    from langsmith.schemas import Example, Run


@pytest.fixture
def run() -> Run:
    from langsmith.schemas import Run

    return Run(
        id=UUID("f77cd087-48f7-4c62-9e0e-297842202107"),
        name="My Run",
        inputs={"input": "What is the answer to life, the universe, and everything?"},
        outputs={"output": "The answer is 42."},
        start_time="2021-07-20T15:00:00.000000+00:00",
        end_time="2021-07-20T15:00:00.000000+00:00",
        run_type="chain",
        execution_order=1,
    )


@pytest.fixture
def example() -> Example:
    from langsmith.schemas import Example

    return Example(
        id=UUID("f77cd087-48f7-4c62-9e0e-297842202106"),
        dataset_id=UUID("f77cd087-48f7-4c62-9e0e-297842202105"),
        inputs={"input": "What is the answer to life, the universe, and everything?"},
        outputs={"output": "The answer is 42."},
        created_at="2021-07-20T15:00:00.000000+00:00",
    )


@pytest.mark.requires("langsmith")
def test_get_qa_evaluator(run: Run, example: Example) -> None:
    """Test get_qa_evaluator."""
    eval_llm = FakeLLM(
        queries={"a": "This checks out.\nCORRECT"}, sequential_responses=True
    )
    qa_evaluator = get_qa_evaluator(eval_llm)
    res = qa_evaluator.evaluate_run(run, example)
    assert res.value == "CORRECT"
    assert res.score == 1


@pytest.mark.requires("langsmith")
def test_get_criteria_evaluator(run: Run, example: Example) -> None:
    """Get a criteria evaluator."""
    eval_llm = FakeLLM(queries={"a": "This checks out.\nY"}, sequential_responses=True)
    criteria_evaluator = get_criteria_evaluator(eval_llm, criteria="conciseness")
    res = criteria_evaluator.evaluate_run(run, example)
    assert res.value == "Y"
    assert res.score == 1
