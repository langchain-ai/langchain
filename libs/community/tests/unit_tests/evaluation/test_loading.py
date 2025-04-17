"""Test the loading function for evaluators."""

from typing import List

import pytest
from langchain.evaluation.loading import EvaluatorType, load_evaluators
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain_core.embeddings import FakeEmbeddings

from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.requires("rapidfuzz")
@pytest.mark.parametrize("evaluator_type", EvaluatorType)
def test_load_evaluators(evaluator_type: EvaluatorType) -> None:
    """Test loading evaluators."""
    fake_llm = FakeChatModel()
    embeddings = FakeEmbeddings(size=32)
    load_evaluators([evaluator_type], llm=fake_llm, embeddings=embeddings)

    # Test as string
    load_evaluators(
        [evaluator_type.value],  # type: ignore[list-item]
        llm=fake_llm,
        embeddings=embeddings,
    )


@pytest.mark.parametrize(
    "evaluator_types",
    [
        [EvaluatorType.LABELED_CRITERIA],
        [EvaluatorType.LABELED_PAIRWISE_STRING],
        [EvaluatorType.LABELED_SCORE_STRING],
        [EvaluatorType.QA],
        [EvaluatorType.CONTEXT_QA],
        [EvaluatorType.COT_QA],
        [EvaluatorType.COT_QA, EvaluatorType.LABELED_CRITERIA],
        [
            EvaluatorType.COT_QA,
            EvaluatorType.LABELED_CRITERIA,
            EvaluatorType.LABELED_PAIRWISE_STRING,
        ],
        [EvaluatorType.JSON_EQUALITY],
        [EvaluatorType.EXACT_MATCH, EvaluatorType.REGEX_MATCH],
    ],
)
def test_eval_chain_requires_references(evaluator_types: List[EvaluatorType]) -> None:
    """Test loading evaluators."""
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"}, sequential_responses=True
    )
    evaluators = load_evaluators(
        evaluator_types,
        llm=fake_llm,
    )
    for evaluator in evaluators:
        if not isinstance(evaluator, (StringEvaluator, PairwiseStringEvaluator)):
            raise ValueError("Evaluator is not a [pairwise]string evaluator")
        assert evaluator.requires_reference
