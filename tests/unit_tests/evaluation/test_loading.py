"""Test the loading function for evalutors."""

import pytest

from langchain.evaluation.loading import EvaluatorType, load_evaluators
from tests.unit_tests.llms.fake_chat_model import FakeChatModel


@pytest.mark.parametrize("evaluator_type", EvaluatorType)
def test_load_evaluators(evaluator_type: EvaluatorType) -> None:
    """Test loading evaluators."""
    fake_llm = FakeChatModel()
    load_evaluators([evaluator_type], llm=fake_llm)

    # Test as string
    load_evaluators([evaluator_type.value], llm=fake_llm)  # type: ignore
