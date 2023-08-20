"""Tests for the string run evaluator."""

from unittest.mock import MagicMock

from langchain.evaluation import criteria
from langchain.smith.evaluation.string_run_evaluator import (
    ChainStringRunMapper,
    StringRunEvaluatorChain,
)
from tests.unit_tests.llms import fake_llm


def test_evaluate_run() -> None:
    run_mapper = ChainStringRunMapper()
    example_mapper = MagicMock()
    string_evaluator = criteria.CriteriaEvalChain.from_llm(fake_llm.FakeLLM())
    evaluator = StringRunEvaluatorChain(
        run_mapper=run_mapper,
        example_mapper=example_mapper,
        name="test_evaluator",
        string_evaluator=string_evaluator,
    )
    run = MagicMock()
    example = MagicMock()
    res = evaluator.evaluate_run(run, example)
    assert res.comment.startswith("Error evaluating run ")
    assert res.key == string_evaluator.evaluation_name
