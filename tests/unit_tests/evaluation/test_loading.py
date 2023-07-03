"""Test the loading function for evalutors."""

import pytest

from langchain.evaluation.loading import EvaluatorType, load_evaluators
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.parametrize("evaluator_type", EvaluatorType)
def test_load_evaluators(evaluator_type: EvaluatorType) -> None:
    """Test loading evaluators."""
    fake_llm = FakeChatModel()
    load_evaluators([evaluator_type], llm=fake_llm)

    # Test as string
    load_evaluators([evaluator_type.value], llm=fake_llm)  # type: ignore


def test_criteria_eval_chain_requires_reference() -> None:
    """Test loading evaluators."""
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"}, sequential_responses=True
    )
    evaluator = load_evaluators(
        [EvaluatorType.CRITERIA], llm=fake_llm, requires_reference=True
    )[0]
    if not isinstance(evaluator, StringEvaluator):
        raise ValueError("Evaluator is not a string evaluator")
    assert evaluator.requires_reference
