"""Test the loading function for evalutors."""

import pytest

from langchain.embeddings.fake import FakeEmbeddings
from langchain.evaluation.loading import EvaluatorType, load_evaluators
from tests.unit_tests.llms.fake_chat_model import FakeChatModel


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("evaluator_type", EvaluatorType)
def test_load_evaluators(evaluator_type: EvaluatorType) -> None:
    """Test loading evaluators."""
    fake_llm = FakeChatModel()
    embeddings = FakeEmbeddings(size=32)
    load_evaluators([evaluator_type], llm=fake_llm, embeddings=embeddings)

    # Test as string
    load_evaluators(
        [evaluator_type.value],  # type: ignore
        llm=fake_llm,
        embeddings=embeddings,
    )
